# Image-classification fine-tuned ResNet50
Use fine-tuned ResNet50 to classify brain tumors. ResNet50 is pre-trained on ImageNet. The fully connected layer of the model is changed. DropOut is used in the fully connected layer to prevent overfitting. In addition, the loss function uses cross entropy, which is added Label Smoothing.The project can realize the category prediction of a single picture. This is just a small exercise for classification and prediction tasks, there are many shortcomings, welcome to discuss 

## Dataset structure
* Brain Tumor Classification
  * train
    * glioma_tumor
      * 001.jpg
      * 002.jpg
      * ...
    * meningioma_tumor
    * no_tumor
    * pituitary_tumor
  * val
    * glioma_tumor  
    * meningioma_tumor
    * no_tumor
    * pituitary_tumor

## Project structure
* loss_acc: Store the accuracy and loss curve generated after the training and verification process.
* models: Store the model when the last training is completed.
* test: Picture for prediction.
* generate_val.py: If you only have a training set, you can use it to generate a validation set by extracting a portion of the images according to the ratio. A ratio of 0.2 is recommended as the validation set.
* labelsmoothing.py: Add Label Smoothing to the loss function to prevent overfitting.
* net.py: Define the model. The file contains the fine-tuned VGG16 and ResNet50 pre-trained models. You only need to change the number of final output categories.
* train_val.py: Training and validation.
* predict.py: Input a picture, it will tell you the category of the target in the picture.






