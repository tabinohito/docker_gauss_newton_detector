# Docker_gauss_newton_detector
## Build Docker
- Normal case
```
sudo ./build.sh
```

- To avoid having to use sudo when you use the docker command
```
./build.sh
```

## Run Docker
- Normal case
```
sudo ./run.sh
```

- To avoid having to use sudo when you use the docker command
```
./run.sh
```

## Run similarity_transformation
```
cd /workspace/src/similarity_transformation/build
cmake --build .
./main <theta param> <scale param> <Input image name>.png 
```

## Run gauss_newton_detector
```
cd /workspace/src/gauss_newton_detector/build
cmake --build .
./main <Input image name>.png <Input image name>_Similarity.png <<option>theta> <<option> scale>
```
