# Self Supervised Learning for Anomaly Detection
This repository contains a self-supervised learning routine to allow for the automated clustering of images together. This has been used for anomaly detection by deletion of the clusters which have the smallest number of images. 

Future work will involve a GUI that will allow a user to specify images that are considered good examples of the image class. This allows a curriculum to be applied which uses the cosine similarity between the selected references and all other images. It has been found empirically that this improves the quality of the clusters generated.
