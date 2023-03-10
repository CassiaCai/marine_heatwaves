*This is under development.*

# objMeasures

![pytest workflow](https://github.com/CassiaCai/marine_heatwaves/actions/workflows/pytest.yml/badge.svg)

We built **objMeasures** to enable users to calculate multiple object tracking measures. **objMeasures** first developed as an extension of the morphological image processing Python package [Ocetrac](https://github.com/ocetrac/ocetrac), where we worked with tracked unique geospatial features (i.e., sea surface temperature extremes, particle tracer tracks, and chlorophyll-a concentrations among others) in gridded datasets, such as climate model simulations, reanalysis datasets, and observations. The motivation is to compare characteristics between multi-dimensional objects and to compute distances between the objects to understand similarities and streamline the data analysis. 

### Use case: marine heatwaves
We apply **objMeasures** on sea surface temperature anomalies (SSTAs) in CESM Large Ensemble (CESM-LE) climate model projections, which consists of 100 ensemble members at 1° spatial resolution covering 1850 to 2100. Our multi-dimensional SSTA objects are called marine heatwaves (MHWs). Classifying and finding patterns in the spatiotemporal evolution of these moving SST ‘groupings’ can fill multiple knowledge gaps, such as our understanding of key MHW characteristics like distribution, variability, and trends, and the physical mechanisms that cause MHWs in different parts of the ocean. We can then generate statistics from our MHW groups, which will allow us to examine the global and regional scale drivers of MHWs. This pipeline allows this analysis to be replicated for (1) different regions and (2) using different datasets. 

Scientifically, this project will add merit to the field by allowing us to 1) assess the possible physical processes causing MHWs, 2) to define the the statistical properties of MHWs, including their varying intensities, duration, and spatial extents and 3) to characterize the spatio-evolution of connected MHWs. More generally, this project will advance our mechanistic understanding of the MHWs, relate one MHW to another, and our understanding of MHW evolution.

<div align="center">
<img width="563" alt="Screen Shot 2023-03-09 at 3 16 35 PM" src="https://user-images.githubusercontent.com/52092892/224192104-45acd60f-f071-4b9c-9173-edd3e1c2377e.png">
</div>
<div align="center">
Figure 1. Three-dimensional visualization of marine heatwaves over time
</div>

<div align="center">
  <a href="https://raw.githubusercontent.com/CassiaCai/marine_heatwaves/main/figures/threedviz.html">
    Click for interactive visualization
  </a>
</div>

## Installation

## How you can contribute
- You can get involved by trying objMeasures, filing [issues](https://github.com/CassiaCai/marine_heatwaves/issues) if you find problems, and making [pull requests](https://github.com/CassiaCai/marine_heatwaves/pulls) if you make improvements.

## Wishlist

## Acknowledgements
- This work grew from a collaboration with the eScience Institute at UW as part of the 2023 Winter Incubator Program. This project is supported by (1) [NSF OCE 2022874](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2022874&HistoricalAwards=false), (2) the [UW Program on Climate Change Graubard Research Acceleration Fund](https://pcc.uw.edu/research/funding-opportunities/), and (3) the [UW Leo Cup](https://www.ocean.washington.edu/story/Leo_Cup).
