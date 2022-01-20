#!/bin/bash 
wget -O jaspar_pfm/pfm.zip https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_non-redundant_pfms_jaspar.zip
unzip -d jaspar_pfm JASPAR2022_CORE_non-redundant_pfms_jaspar.zip
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
wget https://jaspar.genereg.net/download/database/JASPAR2022.sql.gz
gunzip -c JASPAR2022.sql.gz | sqlite3 JASPAR2022.sqlite3
sqlite3 JASPAR2022.sqlite3 "SELECT * FROM MATRIX ;" > jaspar_self_ids.txt
sqlite3 JASPAR2022.sqlite3 "SELECT * FROM MATRIX_PROTEIN;" > jaspar_uniprot_ids.txt
