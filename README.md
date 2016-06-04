# TSP NN Training
This repository is one of three parts that combines into one bigger project that resolves Travelling Salesman Problem using recurrent neural network.  

## Technologies
Used technologies:
- Python 3.4
- PyBrain

## Usage
Make sure that you're using Python 3.4 and you've got PyBrain installed.  
To train neural network run `main.py` script:
```
python3 -m main
```
Script will generate many neural networks (in `networks` subfolder) that have been saved in 30 seconds interval between training. To choose the best one use another script:
```
python3 -m testing
```
It will check average percentage difference between correct TSP solution and NN solution. The best network will be printed out at the end.

## Other project repositories
- [TSP NN UI](https://github.com/jpowie01/TSP-NN-UI)
- [TSP NN API](https://github.com/jpowie01/TSP-NN-API)

## Authors
- [Jakub Powierza](https://github.com/jpowie01)
- [Karolina Olszewska](https://github.com/kolszewska)
