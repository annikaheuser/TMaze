# TMaze
The Google Colabatory notebook, [Demo.ipynb](https://github.com/annikaheuser/TMaze/blob/main/Demo.ipynb), demonstrates how we produced the experimental materials for our validation experiment, based on those from [Boyce et al. 2020](https://www.sciencedirect.com/science/article/pii/S0749596X19301147?casa_token=dagGMwp31pkAAAAA:oz8XctgyhvYLYS0WApWTnWTwg-90SqG9ayjfnSRniwsgxQ7dCieP1jAfQO6xVt1u2ed-kqXBNg) More details about the experiment, as well as the implementation of TMaze, can be found in [Heuser, 2022](https://dspace.mit.edu/handle/1721.1/147233). 

The package from which we sourced the large lanaguage model, [mlm-scoring](https://github.com/awslabs/mlm-scoring), requires CUDA, which Google Colab has pre-installed. Therefore, anyone, no matter their hardware, can run that notebook on Google Colab. Special thanks to Kyle Vedder for help in setting up the notebook environment to be compatible with the mlm-scoring package. 

If the imports from the .py files fail, we recommend restarting the runtime first. If it continues to fail in the same way, consider using [CodeIncludedDemo.ipynb](https://github.com/annikaheuser/TMaze/blob/main/CodeIncludedDemo.ipynb) instead.

Collaborations to improve this codebase (and therefore also pull requests) are welcome, as are inquiries about adjusting TMaze to your experiment. 

