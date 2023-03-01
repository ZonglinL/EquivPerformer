

Command this

```
python3 pccls_run.py --num_random 20 --num_points 256 --batch_size 16 --head 8 --num_channels 8 --lr 3e-3
```

This is for reload
```
--restore ./trained/pc3d_dgl.pt
```
remove selfint
```
--siend 1x1  
--simid 1x1  
```

 add selfint
```
--siend att 
--simid att 
```


 use performer
 adjust max number of random features and antithetic or not
```
--kernel --num_random 8 --antithetic
```
