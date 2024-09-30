#!/usr/bin/env nextflow

// Define the WSI input file
params.wsi = "/home/ubuntu/bala/bala/ImpartLabs/tmp/DI_dombox2_0006.svs"

// Define the output directory for results
params.outdir = "results"

// Create the output directory if it doesn't exist
new File(params.outdir).mkdirs()

// Define the input channel for the WSI file
Channel.fromPath(params.wsi).set { wsi_file }

// Process: read_wsi
process read_wsi {
    input:
        path wsi_file

    output:
        path "${params.outdir}/thumbnail.png", emit: wsi_thumbnail

    script:
    """
    python Scripts/read_wsi.py --input $wsi_file --output ${params.outdir}/thumbnail.png
    """
}

// Process: stain_normalization
process stain_normalization {
    input:
        path wsi_file

    output:
        path "${params.outdir}/normalized_wsi.png", emit: normalized_wsi

    script:
    """
    python Scripts/stain_normalization.py --input $wsi_file --output ${params.outdir}/normalized_wsi.png
    """
}

// Process: tissue_mask
process tissue_mask {
    input:
        path normalized_wsi

    output:
        path "${params.outdir}/tissue_mask.png", emit: tissue_mask

    script:
    """
    python Scripts/tissue_mask.py --input $normalized_wsi --output ${params.outdir}/tissue_mask.png
    """
}

// Process: nuclei_segmentation
process nuclei_segmentation {
    input:
        path normalized_wsi
        path tissue_mask

    output:
        path "${params.outdir}/nuclei_result.pkl", emit: nuclei_result

    script:
    """
    python Scripts/hovernet.py --input $normalized_wsi --mask $tissue_mask --output ${params.outdir}/nuclei_result.pkl
    """
}

// Process: feature_extraction
process feature_extraction {
    input:
        path nuclei_result

    output:
        path "${params.outdir}/features.csv", emit: extracted_features

    script:
    """
    python Scripts/feature_extract.py --input $nuclei_result --output ${params.outdir}/features.csv
    """
}

// Process: model_inference
process model_inference {
    input:
        path extracted_features

    output:
        path "${params.outdir}/prediction.txt", emit: prediction

    script:
    """
    python Scripts/model_inference.py --input $extracted_features --output ${params.outdir}/prediction.txt
    """
}

// Process: visualize_heatmap
process visualize_heatmap {
    input:
        path normalized_wsi
        path prediction

    output:
        path "${params.outdir}/heatmap.png", emit: heatmap

    script:
    """
    python Scripts/visualize_heatmap.py --input $normalized_wsi --prediction $prediction --output ${params.outdir}/heatmap.png
    """
}

// Process: extract_tiles
process extract_tiles {
    input:
        path normalized_wsi
        path heatmap

    output:
        path "${params.outdir}/tiles/*", emit: tiles

    script:
    """
    python Scripts/extract_tiles.py --input $normalized_wsi --heatmap $heatmap --output ${params.outdir}/tiles/
    """
}

// Workflow Definition
workflow {
    // Start with the WSI file
    read_wsi(wsi_file)

    // Perform stain normalization
    stain_normalization(wsi_file)

    // Create tissue mask using normalized WSI
    tissue_mask(stain_normalization.out.normalized_wsi)

    // Perform nuclei segmentation
    nuclei_segmentation(
        stain_normalization.out.normalized_wsi,
        tissue_mask.out.tissue_mask
    )

    // Extract features from nuclei segmentation
    feature_extraction(nuclei_segmentation.out.nuclei_result)

    // Perform model inference
    model_inference(feature_extraction.out.extracted_features)

    // Visualize heatmap
    visualize_heatmap(
        stain_normalization.out.normalized_wsi,
        model_inference.out.prediction
    )

    // Extract tiles
    extract_tiles(
        stain_normalization.out.normalized_wsi,
        visualize_heatmap.out.heatmap
    )
}
