<a name="readme-top"></a>

<!-- TITLE -->
<div align="center">
  <h3 align="center">Measuring the Complexity of Short Swaps using Neural Networks</h3>
  Mark Kevn A. Ong Yiu · David Demitri Africa · Carlo Gabriel Pastor
  <br />
  <br />
  <p align="center">
    A capstone project in partial fullfillment of the Masters in Data Science course at the Ateneo de Manila University.
    <br />
    <a href="https://github.com/ongyiumark/analyzing-short-swaps"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ongyiumark/analyzing-short-swaps">View Demo</a>
    ·
    <a href="https://github.com/ongyiumark/analyzing-short-swaps/issues">Report Bug</a>
    ·
    <a href="https://github.com/ongyiumark/analyzing-short-swaps/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributors">Contributors</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE -->
## Usage

1. Compile the generator into a binary
    ```sh
    g++ swap-gen.cpp -O3 -o swap-gen
    ```
2. Run the binary
    ```sh
    ./swap-gen N M [DIR]
    ```
    where `N` is the length of the permutation, and `M` is the bound for swaps. For example, adjacent swap is `M=2` and short swap is `M=3`.

    `DIR` is an optional name for the subdirectory where the data will be saved. By default, the subdirectory is `[M]swap`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap
- [x] Generate data for various swapping bounds
- [ ] Train neural network to predict the next move
- [ ] Create pipeline to train neural network

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTORS -->
## Contributors
- [Mark Kevin A. Ong Yiu](https://github.com/ongyiumark)

<p align="right">(<a href="#readme-top">back to top</a>)</p>