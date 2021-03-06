# captcha_recognition
This repository uses crnn to recognize capcha and an average pooling is added so that the input height can change to 64.



# Dependencies
* python3
* opencv-python
* torch
* torchvision
* numpy

# Run test
```bash
python3 infer.py --model_path crnn_capcha.pth --imgs_dir imgs
```


# Result
Test on python generated 10000 images
```bash
elapsed time: 25.085253953933716, accuracy: 0.9523
```
| Ground truth 	| Prediction 	| Image 	|
|--------------	|------------	|-------	|
| gPKuG 	| gPKug 	| ![1](imgs/0.png "1") 	|
| izyP 	| izyP 	| ![2](imgs/1.png "2") 	|
| fM7n | fM7n |  ![3](imgs/2.png "3")	|
| txjA | txjA |  ![4](imgs/3.png "4")	|
| LWCN | lWCN |  ![5](imgs/4.png "5")	|
| fa0Z | fa0Z |  ![6](imgs/5.png "6")	|
| PKqOE | PKqOE | ![7](imgs/6.png "7") 	|
| AkkHB | AkkHB | ![8](imgs/7.png "8") 	|
| owMj | owMj |  ![9](imgs/8.png "9")	|
| rtmXI | rtmXI | ![10](imgs/9.png "10") 	|
| Ox2wG | Ox2wG |  ![11](imgs/10.png "11")	|
| sL82v | sl82v | ![12](imgs/11.png "12") 	|
| ncOy3 | nc0y3 | ![13](imgs/12.png "13") 	|
| PLjz | PLjz |  ![14](imgs/13.png "14")	|
| mU0Na | mU0Na | ![15](imgs/14.png "15") 	|


# Analysis
Now the recognize errors are just some characters confuse, like 'x' and 'X', 'p' and 'P', 'o' and 'O'