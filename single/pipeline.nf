#!/usr/bin/env nextflow

// Define parameters
params.wsi = "/home/ubuntu/bala/bala/ImpartLabs/tmp/DI_dombox2_0006.svs"
params.outdir = "/home/ubuntu/bala/bala/ImpartLabs/tmp/results"
params.scripts = "/home/ubuntu/bala/bala/ImpartLabs/TIA_Pipeline/single/Scripts"

// Create the output directory if it doesn't exist
new File(params.outdir).mkdirs()

// Define the input channel for the WSI file
Channel.fromPath(params.wsi).set { wsi_file }

// Process: read_wsi
process read_wsi {
    input:
        path wsi_file

    output:
        path "thumbnail.png", emit: wsi_thumbnail

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/read_wsi.py --input $wsi_file --output thumbnail.png
    """
    publishDir "${params.outdir}", mode: 'copy'
}

// Process: stain_normalization
process stain_normalization {
    input:
        path wsi_file

    output:
        path "normalized_wsi.png", emit: normalized_wsi

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/stain_normalization.py --input $wsi_file --output normalized_wsi.png
    """
    publishDir "${params.outdir}", mode: 'copy'
}

// Process: tissue_mask
process tissue_mask {
    input:
        path normalized_wsi

    output:
        path "tissue_mask.png", emit: tissue_mask

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/tissue_mask.py --input $normalized_wsi --output tissue_mask.png
    """
    publishDir "${params.outdir}", mode: 'copy'
}

// Process: nuclei_segmentation
process nuclei_segmentation {
    input:
        path normalized_wsi
        path tissue_mask

    output:
        path "nuclei_result.pkl", emit: nuclei_result

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/hovernet.py --input $normalized_wsi --mask $tissue_mask --output nuclei_result.pkl
    """
    publishDir "${params.outdir}", mode: 'copy'
}

// Process: feature_extraction
process feature_extraction {
    input:
        path nuclei_result

    output:
        path "features.csv", emit: extracted_features

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/feature_extract.py --input $nuclei_result --output features.csv
    """
    publishDir "${params.outdir}", mode: 'copy'
}

// Process: model_inference
process model_inference {
    input:
        path extracted_features

    output:
        path "prediction.txt", emit: prediction

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/model_inference.py --input $extracted_features --output prediction.txt
    """
    publishDir "${params.outdir}", mode: 'copy'
}

// Process: visualize_heatmap
process visualize_heatmap {
    input:
        path normalized_wsi
        path prediction

    output:
        path "heatmap.png", emit: heatmap

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/visualize_heatmap.py --input $normalized_wsi --prediction $prediction --output heatmap.png
    """
    publishDir "${params.outdir}", mode: 'copy'
}

// Process: extract_tiles
process extract_tiles {
    input:
        path normalized_wsi
        path heatmap

    output:
        path "tiles/", emit: tiles_dir

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    mkdir tiles
    python ${params.scripts}/extract_tiles.py --input $normalized_wsi --heatmap $heatmap --output tiles/
    """
    publishDir "${params.outdir}", mode: 'copy'
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
