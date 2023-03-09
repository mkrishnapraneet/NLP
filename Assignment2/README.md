# Instructions on how to run :

In the terminal, run:
<!-- code snippet -->
```$ python3 pos_tagger.py```

There exists a pre-trained model in the folder, named ```model.pt```.
Upon running the ```pos_tagger.py``` python file, a prompt will appear asking whether you want to load the pre-trained model, or train and save a new one. Select the appropriate option by typing ```load``` or ```train```.
Depending on the option selected, the program will either load the pre-trained model, or train a new one and save it in the folder. It will also display the accuracy of the model for train/validation/test splits. It also diplays in a tabled format, the accuracy, precision, recall and F1 score for each POS tag.

The program will then prompt you to enter a sentence. Enter the sentence and press enter. The program will then display the POS tags for each word in the sentence.

