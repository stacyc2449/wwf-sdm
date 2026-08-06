# Species Distribution Modelling of common pests
Project looks at most detrimental pests (measured by bushels lost between 2022 through 2024) of corn, soy, and wheat in the U.S.

### Using this repository
This repository features the codebase and final output data, for storage reasons.
In order to obtain input data, download it from [here]()

### Notes/Discussion
1. We have chosen pests that have caused more than 1 million bushels lost in 2022, 2023, and 2024. Some of these listed pests from [Crop Protection Network](https://cropprotectionnetwork.org/) are more general clades; we distinguished between them as much as possible and isolated the most detrimental species in these groups

2. Some insecticides may be effective but not tested against certain species, or commercially advertised to.

3. We are unable to account for how natural plant pest resistance is affected by changing climate, although it is a substantial factor. Some plants seem to have weakened resistance in warmer climates, which may also correspond to an even larger pest pressure in a warmer climate

4. Overwintering conditions were difficult to isolate (we look at this for NST efficacy), in addition to general ranges. Some publications test for the survivability of insects in extreme limits of overwintering conditions, but future prediction data is not precise and accurate enough to predict a one-two days of extreme freeze. Because of this, we determine the ranges of these pests by sightings, accumulated from GBIF which includes samples from universities, museums, and iNaturalist. We filter for winter months and early-season planting to isolate the "overwintering" ranges for these pests.   
We choose to only stick with GBIF datasets because they were the most extensive, and already include large datasets such as iNaturalist sightings

5. We use climactic variables from 1980 and onwards, specifically ones that would be related to overwintering conditions in North America

6. We should put more emphasis on southern-residing pests, since these are most likely to spread to the midwest, with the most crop production

### Goals
The goal is to predict the new expanded range of overwintering sites for common pests, and compare it with areas where neonics application is largely unregulated and legalized. This will provide new recommended places where neonics should be, and shouldn't be applied, because of the detriment that neonics contribute to the environment, such as pollinator death and decreased biodiversity, as well as making it more likely for common pests to develop a resistance to these insecticides.

It is also important to note that neonics may not even be necessary - multiple studies have shown little to no difference in crop yield with applications of seed treatments especially in northern regions. Read more from [WWF](https://www.worldwildlife.org/publications/neonicotinoid-seed-treatments-in-north-american-row-crops-a-literature-review-of-yield-and-profitability-outcomes/).




Updated August 6, 2026