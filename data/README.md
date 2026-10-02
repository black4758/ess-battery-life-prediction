# ESS Battery Project Dataset

## Dataset Source
- **Origin**: Severson et al., *"Data-driven prediction of battery cycle life before capacity degradation"*, Nature Energy (2019).
- **Target Cells**: Commercial A123 Systems APR18650M1A LFP/graphite cells (Nominal capacity: 1.1 Ah, Nominal voltage: 3.3 V).

## File Descriptions
- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`: **Batch 1** (Train set, 46 valid cells, 44 different charging protocols).
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`: **Batch 2** (Primary test set, 39 valid cells, fast charging test).
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`: **Batch 3** (Secondary test set, 44 valid cells, improved cooling fixture `newstructure`).
- `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat`: Variable charging test dataset.

> **Note**: Due to GitHub file size limits (>100MB), `.mat` raw data files are excluded via `.gitignore`.
