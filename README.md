# Anomaly Detection on Subway Operational Data 
- This is my intern project working in CASCO Shanghai, China, Summer 2018. 
- This project is implemented and modified based on the python anomaly detection tooklit [pyod]. 
- I created a new model called cascoKNN to process CASCO subway operational indicator data. 
 
![](/examples/KNN.png)

## Installation 
For more information, check out [pyod].
```sh
git clone https://github.com/paragon520/Pyod_CASCO 
cd Pyod_CASCO
python setup.py install

``` 
## Required Dependency
- Python 2.7, 3.4, 3.5 or 3.6
- numpy>=1.13
- scipy>=0.19.1
- scikit_learn>=0.19.1
- matplotlib                      
- nose   

## Test and run  

Simply run: 
```sh
cd example
python cascoKNN_example.py
```

## View your result 
- ROC value and & rank will be printed on the cmd.
- You can also save the scatter plots to png file.
 
## Credits
All the credits goes to  [yzhao062] . Thanks for providing such a nice toolkit for anomaly detection model.

 [pyod]: https://github.com/yzhao062/Pyod
[yzhao062]: https://github.com/yzhao062/Pyod
