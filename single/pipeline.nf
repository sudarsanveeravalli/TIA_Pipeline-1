#!/usr/bin/env nextflow

// Define the WSI input file
params.wsi = "data/example.svs"

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
    path "${params.outdir}/thumbnail.png" into wsi_thumbnail

    script:
    """
    python scripts/read_wsi.py --input $wsi_file --output ${params.outdir}/thumbnail.png
    """
}

// Process: stain_normalization
process stain_normalization {
    input:
    path wsi_file

    output:
    path "${params.outdir}/normalized_wsi.png" into normalized_wsi

    script:
    """
    python scripts/stain_normalization.py --input $wsi_file --output ${params.outdir}/normalized_wsi.png
    """
}

// Process: tissue_mask
process tissue_mask {
    input:
    path normalized_wsi

    output:
    path "${params.outdir}/tissue_mask.png" into tissue_mask

    script:
    """
    python scripts/tissue_mask.py --input $normalized_wsi --output ${params.outdir}/tissue_mask.png
    """
}

// Process: nuclei_segmentation
process nuclei_segmentation {
    input:
    path normalized_wsi
    path tissue_mask

    output:
    path "${params.outdir}/nuclei_result.pkl" into nuclei_result

    script:
    """
    python scripts/hovernet.py --input $normalized_wsi --mask $tissue_mask --output ${params.outdir}/nuclei_result.pkl
    """
}

// Process: feature_extraction
process feature_extraction {
    input:
    path nuclei_result

    output:
    path "${params.outdir}/features.csv" into extracted_features

    script:
    """
    python scripts/feature_extract.py --input $nuclei_result --output ${params.outdir}/features.csv
    """
}

// Process: model_inference
process model_inference {
    input:
    path extracted_features

    output:
    path "${params.outdir}/prediction.txt" into prediction

    script:
    """
    python scripts/model_inference.py --input $extracted_features --output ${params.outdir}/prediction.txt
    """
}

// Process: visualize_heatmap
process visualize_heatmap {
    input:
    path normalized_wsi
    path prediction

    output:
    path "${params.outdir}/heatmap.png" into heatmap

    script:
    """
    python scripts/visualize_heatmap.py --input $normalized_wsi --prediction $prediction --output ${params.outdir}/heatmap.png
    """
}

// Process: extract_tiles
process extract_tiles {
    input:
    path normalized_wsi
    path heatmap

    output:
    path "${params.outdir}/tiles/*" into tiles

    script:
    """
    python scripts/extract_tiles.py --input $normalized_wsi --heatmap $heatmap --output ${params.outdir}/tiles/
    """
}

// Workflow Definition
workflow {
    // Start with the WSI file
    read_wsi(wsi_file)

    // Connect the outputs to the next processes
    stain_normalization(wsi_file)

    // Proceed with the normalized WSI
    tissue_mask(normalized_wsi)
    nuclei_segmentation(normalized_wsi, tissue_mask)
    feature_extraction(nuclei_result)
    model_inference(extracted_features)
    visualize_heatmap(normalized_wsi, prediction)
    extract_tiles(normalized_wsi, heatmap)
}
