#!/usr/bin/env nextflow

// Define parameters
params.input = "/home/ubuntu/bala/bala/ImpartLabs/tmp/input/sample_small.svs"
params.outdir = "/home/ubuntu/bala/bala/ImpartLabs/tmp/results/"
params.scripts = "/home/ubuntu/bala/bala/ImpartLabs/TIA_Pipeline/single/Scripts"

// Create the output directory if it doesn't exist
new File(params.outdir).mkdirs()

// Define the input channel for the WSI file
Channel.fromPath(params.input).set { wsi_file }

// Process: read_wsi
process read_wsi {
    conda '/path/to/conda/envs/image-processing'
    input:
        path wsi_file

    output:
        path "thumbnail.png", emit: wsi_thumbnail

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/read_wsi.py --input $wsi_file --output thumbnail.png
    """
}

// Process: stain_normalization
process stain_normalization {
    conda '/path/to/conda/envs/image-processing'
    input:
        path wsi_file

    output:
        path "normalized_wsi.png", emit: normalized_wsi

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/stain_normalization.py --input $wsi_file --output normalized_wsi.png
    """
}

// Process: tissue_mask
process tissue_mask {
    conda '/path/to/conda/envs/image-processing'
    input:
        path normalized_wsi

    output:
        path "tissue_mask.png", emit: tissue_mask

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/tissue_mask.py --input $normalized_wsi --output tissue_mask.png
    """
}

// Process: nuclei_segmentation
process nuclei_segmentation {
    conda '/path/to/conda/envs/image-processing'
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
}

// Process: feature_extraction (Optional, if needed)
process feature_extraction {
    conda '/path/to/conda/envs/image-processing'
    input:
        path nuclei_result

    output:
        path "features.csv", emit: extracted_features

    publishDir "${params.outdir}", mode: 'copy'
    script:
    """
    python ${params.scripts}/feature_extract.py --input $nuclei_result --output features.csv
    """
}

// Workflow Definition
workflow {
    // Start with the WSI file
    def wsi_thumbnail = read_wsi(wsi_file)

    // Perform stain normalization
    def norm_wsi = stain_normalization(wsi_thumbnail)

    // Create tissue mask using normalized WSI
    def tissue_mask_im = tissue_mask(norm_wsi)

    // Perform nuclei segmentation
    nuclei_segmentation(
        norm_wsi,
        tissue_mask_im
    )
}
