# Diving into French Presidential Discourses
Research project for the course [Diving into Digital Public Spaces](https://syllabus.sciencespo.fr/cours/202120/195617.html) (Sciences Po, CEU) by Giulia Annaliese Paxton, Ryan Hachem, and Simone Maria Parazzoli. The course was taught by [Jean-Philippe Cointet](https://medialab.sciencespo.fr/equipe/jean-philippe-cointet/) (médialab, Sciences Po) and [Márton Karsai](https://networkdatascience.ceu.edu/people/marton-karsai) (Department of Network and Data Science, Central European University).

## Abstract
The following study aims to elicit the difference in meaning attributed to key words in political discourse, which have the potential to betray latent politics of the candidates in question. The discourses of 7 candidates for the French presidential election of 2022 are analysed by comparing the variable word-embedding models trained on each candidates’ corpora. The corpora are made of the Twitter activity of candidates, the Twitter activity of around 15 affiliates for each candidate, and each candidate’s campaign manifesto. We qualitatively compare the word vectors of key shared words, analysing the most similar words to them in the variable word-embedding spaces to find different political inclinations. We conclude that the difference in political meaning of key concepts is palpable, yet that our method must be paired with a strong qualitative-interpretive approach in order to be analytically and socially relevant. 

## This repository
In this repository you can find the following files: 
- **README.md** - the file you are currently reading
- **paper.md** - an extended description of our research that sets our work withing the literature in political science, data science, and anthropology, that explains its methodological details, and that shows its findings, contributions and limitations
- **code.py** - the Python code that allows interested readers to replicate our research. The code is divided in seven main parts that constitute the sections in which we train the seven models used in our work
- **models** - the folder with the 7 notebooks we wrote to generate the word embeddings models of each of the 7 candidate (look at Emmanuel Macron's notebook for a rich explanation of the script, those of the other candidates' are a replication with with very slight differences) 
- **presentation.pdf** - the slides we used to present our work at Sciences Po on 31/03/22

## Authors
Giulia Annaliese Paxton, Ryan Hachem, and Simone Maria Parazzoli and master students of the School of Public Affairs of Sciences Po. If you want to get in touch, reach us [here](mailto:simoneparazzoli@gmail.com).
