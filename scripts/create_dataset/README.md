# Dataset Creation

## Aim
Create a dataset of proteins related and not with cell cycle for **future analysis**.

---

## Necessary Downloads
Before starting, be sure to have downloaded the following material.

### 1. MSigDB Gene Sets (Human)
Download the entire MSigDB for **Human** in **JSON** format (`Human Gene Set JSON file set (ZIPped)`).
- **Link**: [https://www.gsea-msigdb.org/gsea/downloads.jsp](https://www.gsea-msigdb.org/gsea/downloads.jsp)

### 2. UniProt Human Proteome
Download the list of all **HUMAN proteins** (UniProt human proteome - **UP000005640**).
- **Link (Website)**: [https://www.uniprot.org/proteomes/UP000005640](https://www.uniprot.org/proteomes/UP000005640)
- **Programmatic Download (Terminal)**:

```bash
wget -O human_proteome.tsv.gz "[https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession,reviewed,id,protein_name,gene_names,organism_name,sequence&format=tsv&query=(proteome:UP000005640](https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession,reviewed,id,protein_name,gene_names,organism_name,sequence&format=tsv&query=(proteome:UP000005640))"
gunzip human_proteome.tsv.gz
```

### 3. UniProt Gene-Protein mapping

Both for building Uniref50 cluster and getting protein-gene mapping.

[UniProt protein-gene Mapping](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz)

### 4. Neftel et al. scrna dataset

From 3CA [website](https://www.weizmann.ac.il/sites/3CA/brain)

Download also list of metaporgms (contain cell cyle metaprograms).

```bash
wget https://www.dropbox.com/scl/fi/xflvflx73kbug76u6a955/Data_Neftel2019_Brain.tar.gz?rlkey=o8yru9vryof2ndpa0old3rhn1&dl=1
tar -xvzf 'Data_Neftel2019_Brain.tar.gz?rlkey=o8yru9vryof2ndpa0old3rhn1'


wget https://www.weizmann.ac.il/sites/3CA/sites/sites.3CA/files/meta_programs_2025-01-29.xlsx
```

### 5. InterProt domain

Go to Unirpot and downlaod human rptoeins with the pfa and interprot annotawiton