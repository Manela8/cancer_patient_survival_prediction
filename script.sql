create database cancer_analysis;
use cancer_analysis;

ALTER TABLE cleaned_data RENAME COLUMN `Patient ID` TO patient_id;
ALTER TABLE cleaned_data RENAME COLUMN `Age at Diagnosis` TO age_at_diagnosis;
ALTER TABLE cleaned_data RENAME COLUMN `Type of Breast Surgery` TO type_of_breast_surgery;
ALTER TABLE cleaned_data RENAME COLUMN `Cancer Type Detailed` TO cancer_type_detailed;
ALTER TABLE cleaned_data RENAME COLUMN `Pam50 + Claudin-low subtype` TO pam50_claudinlow_subtype;
ALTER TABLE cleaned_data RENAME COLUMN `ER Status` TO er_status;
ALTER TABLE cleaned_data RENAME COLUMN `Neoplasm Histologic Grade` TO neoplasm_histologic_grade;
ALTER TABLE cleaned_data RENAME COLUMN `HER2 Status` TO her2_status;
ALTER TABLE cleaned_data RENAME COLUMN `Tumor Other Histologic Subtype` TO tumor_other_histologic_subtype;
ALTER TABLE cleaned_data RENAME COLUMN `Hormone Therapy` TO hormone_therapy;
ALTER TABLE cleaned_data RENAME COLUMN `Inferred Menopausal State` TO inferred_menopausal_state;
ALTER TABLE cleaned_data RENAME COLUMN `Integrative Cluster` TO integrative_cluster;
ALTER TABLE cleaned_data RENAME COLUMN `Primary Tumor Laterality` TO primary_tumor_laterality;
ALTER TABLE cleaned_data RENAME COLUMN `Lymph nodes examined positive` TO lymph_nodes_examined_positive;
ALTER TABLE cleaned_data RENAME COLUMN `Mutation Count` TO mutation_count;
ALTER TABLE cleaned_data RENAME COLUMN `Nottingham prognostic index` TO nottingham_prognostic_index;
ALTER TABLE cleaned_data RENAME COLUMN `Oncotree Code` TO oncotree_code;
ALTER TABLE cleaned_data RENAME COLUMN `PR Status` TO pr_status;
ALTER TABLE cleaned_data RENAME COLUMN `Radio Therapy` TO radio_therapy;
ALTER TABLE cleaned_data RENAME COLUMN `3-Gene classifier subtype` TO three_gene_classifier_subtype;
ALTER TABLE cleaned_data RENAME COLUMN `Tumor Size` TO tumor_size;
ALTER TABLE cleaned_data RENAME COLUMN `Tumor Stage` TO tumor_stage;

-- Survival & outcomes

-- Overall survival rate
SELECT
    overall_survival_status,
    COUNT(*) AS patient_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM cleaned_data
GROUP BY overall_survival_status;

-- Survival rate by tumor stage
SELECT
    tumor_stage,
    COUNT(*) AS patients,
    ROUND(100.0 * SUM(CASE WHEN overall_survival_status = 'Living' THEN 1 ELSE 0 END) / COUNT(*), 1) AS survival_rate_pct
FROM cleaned_data
WHERE tumor_stage IS NOT NULL
GROUP BY tumor_stage
ORDER BY tumor_stage;

-- Survival rate by ER/HER2/PR combination
SELECT
    er_status, pr_status, her2_status,
    COUNT(*) AS patients,
    ROUND(100.0 * SUM(CASE WHEN overall_survival_status = 'Living' THEN 1 ELSE 0 END) / COUNT(*), 1) AS survival_rate_pct
FROM cleaned_data
GROUP BY er_status, pr_status, her2_status
ORDER BY patients DESC;

-- Treatment patterns

-- Most common treatment combinations
SELECT
    chemotherapy, hormone_therapy, radio_therapy,
    COUNT(*) AS patients
FROM cleaned_data
GROUP BY chemotherapy, hormone_therapy, radio_therapy
ORDER BY patients DESC;

-- Chemo vs survival, controlled by tumor stage
SELECT
    tumor_stage,
    chemotherapy,
    COUNT(*) AS patients,
    ROUND(100.0 * SUM(CASE WHEN overall_survival_status = 'Living' THEN 1 ELSE 0 END) / COUNT(*), 1) AS survival_rate_pct
FROM cleaned_data
WHERE tumor_stage IS NOT NULL AND chemotherapy IS NOT NULL
GROUP BY tumor_stage, chemotherapy
ORDER BY tumor_stage, chemotherapy;

-- Surgery type vs treatment intensity
SELECT
    type_of_breast_surgery,
    ROUND(AVG(CASE WHEN chemotherapy = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1) AS pct_received_chemo,
    ROUND(AVG(CASE WHEN radio_therapy = 'Yes' THEN 1.0 ELSE 0 END) * 100, 1) AS pct_received_radio,
    COUNT(*) AS patients
FROM cleaned_data
WHERE type_of_breast_surgery IS NOT NULL
GROUP BY type_of_breast_surgery;

-- Tumor & disease characteristics

-- Grade and cellularity together
SELECT
    neoplasm_histologic_grade,
    cellularity,
    COUNT(*) AS patients,
    ROUND(AVG(tumor_size), 1) AS avg_tumor_size_mm
FROM cleaned_data
WHERE neoplasm_histologic_grade IS NOT NULL
GROUP BY neoplasm_histologic_grade, cellularity
ORDER BY neoplasm_histologic_grade;

-- Triple negative segment
SELECT COUNT(*) AS triple_negative_patients,
       ROUND(100.0 * SUM(CASE WHEN overall_survival_status = 'Living' THEN 1 ELSE 0 END) / COUNT(*), 1) AS survival_rate_pct
FROM cleaned_data
WHERE er_status = 'Negative' AND pr_status = 'Negative' AND her2_status = 'Negative';

-- Molecular subtype breakdown
SELECT
    pam50_claudinlow_subtype AS subtype,
    COUNT(*) AS patients,
    ROUND(AVG(mutation_count), 1) AS avg_mutation_count
FROM cleaned_data
GROUP BY pam50_claudinlow_subtype
ORDER BY patients DESC;

-- Demographics & risk segmentation

-- Age-banded outcome view
SELECT
    CASE
        WHEN age_at_diagnosis < 40 THEN 'Under 40'
        WHEN age_at_diagnosis < 50 THEN '40-49'
        WHEN age_at_diagnosis < 60 THEN '50-59'
        WHEN age_at_diagnosis < 70 THEN '60-69'
        ELSE '70+'
    END AS age_band,
    COUNT(*) AS patients,
    ROUND(100.0 * SUM(CASE WHEN overall_survival_status = 'Living' THEN 1 ELSE 0 END) / COUNT(*), 1) AS survival_rate_pct
FROM cleaned_data
GROUP BY 1
ORDER BY MIN(age_at_diagnosis);

-- High-risk cohort
SELECT COUNT(*) AS high_risk_patients
FROM cleaned_data
WHERE inferred_menopausal_state = 'Post'
  AND neoplasm_histologic_grade = 3
  AND lymph_nodes_examined_positive > 0;
  
-- Data quality
SELECT
    ROUND(100.0 * SUM(CASE WHEN chemotherapy IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_missing_chemo,
    ROUND(100.0 * SUM(CASE WHEN her2_status IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_missing_her2,
    ROUND(100.0 * SUM(CASE WHEN cellularity IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_missing_cellularity
FROM cleaned_data; 