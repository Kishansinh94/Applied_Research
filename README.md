The project is created for my module of Applied Research (One month). I built Brain Tumor detection system.
The objectives of project are :<br/>
 - Understand the behaviour of convolutional neural network<br/>
 -To overcome overfitting/underfitiing while working with very small dataset<br/> 
 -Find the good model medical imaging problem<br/>
 
Given below are file descrption: (For more detail and documentation please go through "AI for Medical Sceince.pdf")


# Applied_Research
1. **Tumor.py** file is model which i proposed here new proposed model. Model is based on VGG16. Trained on grayscale images gives better accuraccy than VGG16 and also perform well on unseen data. Six unseen images given out 5 model predicted right.  <br/><br/>
2. **VGG16Tumor.py** is model of VGG16 (not transfer learning). Chnaged parameter and trained on graysclae iamges. <br/><br/>
 Project on Brain tumor detection <br/><br/>
3. **Covid19.py** file contain model which give Training accuracy of 72 percent. Model trained with Xray images with Covid19 positive and other lungs infections. Model intended to indetify Covid19 chect Xray.  Model is not overfiting but having flat tarining accuracy at 72 which is not good.  <br/>  <br/> 
4. **Brain_Tumor.py** model trained on RGB files. This model is overfitted.  <br/><br/>
5. **Callmodelvgg16.py** if file useful to call model in frontend for actual prediction of given images by user. <br/>
