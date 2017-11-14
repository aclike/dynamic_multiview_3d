## To create Gazebo models for a specific synset, say cars
1. Download all of the models from ShapeNet, say in the folder `~Downloads/ShapeNetCore.v2`
2. Obtain the ID for the synset you're interested in, by going to ShapeNet > Browse Taxonomy > (click on synset) > ImageNet > (it should be in the URL now, say as 02958343)
3. `cd` to the top level of the `collect_data` folder: `roscd collect_data`
4. Convert the models to SDF format: `python bin/convert_models.py <synset_name> <shapenet_dir> <gazebo_dir> --force` (say with `python bin/convert_models.py car ~/Downloads/ShapeNetCore.v2/02958343 ~/.gazebo/models --force`)
5. Resize the models: `python bin/resize_models.py <synset_name> <gazebo_dir>` (say with `python bin/resize_models.py car ~/.gazebo/models`)

## To collect data
In one terminal:
```
roscore
```

In another terminal:
```
rosrun collect_data collect_data_node.py <synset name> <desired # models> <desired # pairs per model> <model folder> <desired output folder> --tfr --save_depth --save_rate 300
```
say as `rosrun collect_data collect_data_node.py car 25000 40 ~/.gazebo/models car_dataset --tfr --save_depth --save_rate 300`
