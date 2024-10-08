#!/usr/bin/env nextflow

// Define parameters
params.wsi = "/home/path02/python_envs/ImpartLabs/tmp/input/sample_small.svs"
params.outdir = "/home/path02/python_envs/ImpartLabs/tmp/results/"
params.scripts = "/home/path02/python_envs/ImpartLabs/TIA_Pipeline/single/Scripts"

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
    
}

// Process: feature_extraction

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


}
