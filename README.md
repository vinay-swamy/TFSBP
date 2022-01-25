
## Project

Main goal: predict TF binding affinity *de novo* from protein sequence/ structure 

### Data Sources
[JASPAR](https://jaspar.genereg.net) - curated TF binding site database


### Current implementation

Use a onehot encoded, fixed window representation of AA sequence to predict PWM. PWMs are represented as a fixed window of nt sequence, aligned such that the binding sequence position with the highest information content is in the center of the window. 

The current version of model is a classification based model. A CNN that has a final dense output which is reshaped to the right dimensions of the output, then softmaxed and compared to ground truth via cross entropy. I dont think this is the right loss function however, as its weights the entire sequence equally, when in fact the middle parts are likely more important that the tails. This would be solved with a seq2seq model




Implementation Ideas:
- Seq2Seq model instead of a classification-esque model 
- Use AlphaFold2 model intermediate outputs(similar to [here](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbab564/6509729))