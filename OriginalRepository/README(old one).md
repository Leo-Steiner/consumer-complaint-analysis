# NLP Analysis of Consumer Complaints

## Project overview

This repository contains the development phase of IU course DLBDSEDA02
Task 1. The project uses natural language processing and unsupervised topic
modelling to identify prevalent themes in consumer complaint narratives.

## Dataset

The analysis uses the Consumer Complaint Database available through Kaggle.
The original dataset is not stored in this repository because of its size.

- Dataset: https://www.kaggle.com/datasets/selener/consumer-complaint-database
- Online notebook: https://www.kaggle.com/code/leosteiner0/notebookf59b1fdbbe

A reproducible sample of 10,000 unique, non-empty narratives was
created with random state 42. After preprocessing,
10,000 documents remained.

## Data preparation

The workflow removes missing and duplicate narratives, converts text to
lowercase, removes URLs, email addresses, numbers, punctuation and masked
`XXXX` content, tokenizes the text, removes English stopwords and lemmatizes
the remaining words.

## Vectorization

- CountVectorizer created a Bag-of-Words matrix with
  4,415 features for LDA.
- TF-IDF created a weighted matrix with 4,415 features for
  NMF, reducing the influence of terms that occur in many documents.

The vectorization statistics and highest-ranked terms are stored in the
`results` folder.

## Selecting the number of topics

The topic count was identified through term-centric stability analysis rather
than through subjective inspection. For every candidate from 2 to
20, LDA and NMF were fitted to 5 repeated
80% subsamples. Topics were matched with the Hungarian
algorithm, and ranked top-word agreement was measured with Average Jaccard
similarity.

The highest combined stability was 0.7813. The
defined comparability and diversity rule selected **2
topics**, with combined stability
0.7813.

## Main results

The mean matched LDA-NMF top-word agreement was
0.6960. Supporting C_v coherence was
0.4810 for LDA and 0.5192 for NMF.

### LDA topics

- **LDA topic 2 (Topic 2: payment / loan / account):** payment, loan, account, pay, bank, call, make, get, time, card (58.02% average prevalence)
- **LDA topic 1 (Topic 1: credit / report / account):** credit, report, account, debt, information, dispute, collection, remove, file, letter (41.98% average prevalence)

### NMF topics

- **NMF topic 1 (Topic 1: payment / pay / loan):** payment, pay, loan, call, make, bank, get, time, month, account (56.20% average prevalence)
- **NMF topic 2 (Topic 2: report / credit / account):** report, credit, account, remove, debt, dispute, information, collection, bureau, inquiry (43.80% average prevalence)

## Repository structure

~~~text
README.md
requirements.txt
task1_nlp_analysis.ipynb
results/
    cleaning_summary.csv
    data_quality_summary.csv
    vectorization_comparison.csv
    top_count_terms.csv
    top_tfidf_terms.csv
    model_evaluation.csv
    model_evaluation.png
    topic_words.csv
    lda_topic_prevalence.csv
    lda_topic_prevalence.png
    lda_topic_words.png
    nmf_topic_prevalence.csv
    nmf_topic_prevalence.png
    nmf_topic_words.png
    lda_nmf_topic_comparison.csv
    lda_nmf_topic_overlap.png
    top_products.png
    result_summary.json
~~~

## Run the analysis

1. Open the Kaggle notebook.
2. Attach the Consumer Complaint Database through **Add Input**.
3. Use a standard CPU session.
4. Select **Run All**.
5. Download the completed notebook and `/kaggle/working` outputs.
6. Upload the notebook, README, requirements and results to GitHub.

The notebook is designed to run without internet access. If an optional NLTK
corpus or Gensim is unavailable, a documented fallback is used without
changing the core stability-based selection method.

## Limitations

- Sampling can exclude rare complaint themes.
- Stability measures reproducibility, not guaranteed semantic usefulness.
- Preprocessing choices influence the resulting topic structure.
- Automatic topic labels are transparent summaries of the three
  highest-weighted terms and may require human interpretation.
- The selected topic count applies to this sample and modelling configuration.

## Project links

- GitHub repository: https://github.com/Leo-Steiner/Project-Task-1-iu-nlp-consumer-complaints-
- Kaggle notebook: https://www.kaggle.com/code/leosteiner0/notebookf59b1fdbbe
- Kaggle dataset: https://www.kaggle.com/datasets/selener/consumer-complaint-database

## References

Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent Dirichlet
allocation. *Journal of Machine Learning Research, 3*, 993-1022.

Greene, D., O'Callaghan, D., and Cunningham, P. (2014). How many topics?
Stability analysis for topic models. In *Machine Learning and Knowledge
Discovery in Databases* (pp. 498-513). Springer.

Lee, D. D., and Seung, H. S. (1999). Learning the parts of objects by
non-negative matrix factorization. *Nature, 401*, 788-791.
