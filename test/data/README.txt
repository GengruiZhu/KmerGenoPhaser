KmerGenoPhaser — Test Data Directory
=====================================

This directory contains minimal synthetic data for testing all modules.
Replace with your own real data when running on actual genomes.

Directory structure and expected file formats:
----------------------------------------------

target.fasta
    Small reference genome (2–3 chromosomes, ~200 kb each).
    Used by: unsupervised, supervised

target.genome.size
    Two-column TSV: chrom_name  length_bp
    Example:
        Chr1    200000
        Chr2    200000
    Used by: unsupervised (check_and_fix_blocks), snpml

blocks/
    Per-chromosome block .txt files from upstream modules.
    Format: tab-separated, with header: Start  End  Bloodline
    One file per chromosome, named <ChromName>.txt
    Example Chr1.txt:
        Start   End     Bloodline
        0       1000000 GroupA
        1000000 2000000 GroupB
    Used by: unsupervised (--block_dir)

reads/
    Subdirectory per ancestor group, containing *.fastq.gz files.
    reads/GroupA/sim_GroupA.fastq.gz
    reads/GroupB/sim_GroupB.fastq.gz
    Used by: supervised (--read_dirs)

ad_matrices/
    Per-chromosome AD matrix files named <Chrom>_AD_matrix.txt
    Format: no header, columns = CHROM POS REF ALT <sample1> <sample2> ...
    AD values per sample formatted as "ref_count,alt_count" (e.g. "10,3")
    Used by: snpml (--ad_matrix_dir)

vcf/
    Multi-sample VCF (bgzipped + tabix-indexed).
    test.vcf.gz  +  test.vcf.gz.tbi
    Used by: snpml (--vcf)

group_lists/
    One file per ancestor group, listing sample names (one per line).
    group_lists/GroupA.txt
    group_lists/GroupB.txt
    Used by: snpml (--group_lists)

karyotype_blocks/
    Block .txt files with Bloodline labels for karyotype visualization.
    Same format as blocks/ but may also contain Inferred_* labels.
    Format: tab-separated, header: Start  End  Bloodline  [Chrom]
    Used by: vis_karyotype (--input_dir)

centromeres.csv  (OPTIONAL)
    Centromere position file for accurate idiogram drawing.
    Format: CSV with header:
        Chrom,Centromere_Start_Mb,Centromere_End_Mb
        Chr1A,45.2,48.1
        Chr1B,52.3,55.0
    Used by: vis_karyotype (--centromere_file)
