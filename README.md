## The code is comming soon.

## Description

**Paper:** Junyin Qiu, Jianglin Lan. **Direct Sparse Initialization for Stereo
Visual-Inertial Odometry**, *IEEE Robotics and Automation Letters.* **[PDF](https://ieeexplore.ieee.org/abstract/document/11478329)**.

We propose a direct initialization method for stereo visual-inertial odometry, which directly bridges original image intensities and initial parameters, bypassing conventional intermediate steps such as feature tracking and visual SfM. Extensive experiments confirm that our method achieves superior performance in both estimation accuracy and initialization success rate with shorter runtime. Even with 3 frames for initialization, our method outperforms the state-of-the-art methods using 10 frames in most metrics.

## Requirements

Eigen3
OpenCV 3.4

## Build and Run

```
cd DSInit
mkdir build && cd build
cmake ..
make -j4
```

Modify the dataset_path in the *config/euroc.yaml*
In the build folder, run:
```
./dsinit ../euroc.yaml
```

If you find this work helpful or use our code, please cite:
```bibtex
@ARTICLE{11478329,
  author={Qiu, Junyin and Lan, Jianglin},
  journal={IEEE Robotics and Automation Letters}, 
  title={Direct Sparse Initialization for Stereo Visual-Inertial Odometry}, 
  year={2026},
  volume={11},
  number={6},
  pages={6576-6583}}
```

## License

DSInit is under [GPLv3 license](https://github.com/Qiu0336/DSInit/blob/main/LICENSE).


