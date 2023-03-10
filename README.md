

Command this

```
python3 -W ignore pccls_run.py --num_points 256 --batch_size 64 --head 8 --num_channels 8 --lr 3e-3 --num_layers 5
```

This is for reload, make sure to move your favorite model to directory ```trained```
```
--restore ./trained/pc3d_dgl.pt
```
Linear selfint
```
--siend 1x1  
--simid 1x1  
```

Attentive selfint
```
--siend att 
--simid att 
```


 use performer
 adjust max number of random features and antithetic or not
```
--kernel --num_random 20 --antithetic
```
