# Background on Volumetric Additive Manufacturing
Volumetric Additive Manufacturing in general is about 3D printing arbitrary shapes and objects.
This toolbox solves some of the optimization challenges around TVAM.

## Tomographic Volumetric Additive Manufacturing (TVAM)
Tomographic Volumetric Additive Manufacturing is an emerging 3D printing technology that uses light to create 3D objects.
The technology is based on the principles of tomography, which is a technique for creating 3D images of the internal structures of objects by taking a large number of 2D X-ray images from different angles and then using these images to reconstruct a 3D image of the object.
In TVAM the object is not imaged, but rather the 3D object is created directly from the 2D images which are projected into a volume of photosensitive material.
Of course, this would smear light into all regions of the resin and create a solid block of material.
However, polymerization is inhibited by oxygen or other inhibitors. Hence, the region only polymerizes where the light intensity is high enough to overcome the polymerization threshold.

The process is illustrated in the figure below.
```@raw html
<img src="../assets/principle.png" alt="" width="600"/>
```

### Central Question
The central question in TVAM what are the light pattern such that the desired 3D object is created in the resin?
In other words, void regions should receive less light than polymerized threshold. 
And object regions should receive more light than the polymerized threshold.

Mathematically, we seek to find a set of light patterns $P$ such that the light intensity $I$ in the resin satisfies the following constraints:

$$\mathcal{L} = \underbrace{\sum_{v \,\in\,\text{object}} |\text{ReLu}(T_U - I_v)|^K}_\text{force object polymerization}+\underbrace{\sum_{v\,\notin\,\text{object}} |\text{ReLu}(I_v - T_L) |^K}_{\text{keep empty space unpolymerized}}+$$
$$+\underbrace{\sum_{v \,\in\,\text{object}} |\text{ReLu}(I_v - 1)|^K}_{\text{avoid overpolymerization}} +\underbrace{\sum_{p \,\in\,\text{patterns}} |P_p|^4}_{\text{avoid sparse patterns}}$$

The propagation operator $\mathcal{P}$ can be either a simple ray tracing or a more complex wave propagation model.

$$I_v = \mathcal{P} P$$


### Optimization
The optimization problem is to find the light patterns $P$ that minimize the loss $\mathcal{L}$.
This is a challenging optimization problem, because the light patterns $P$ are high-dimensional and the loss $\mathcal{L}$ is non-convex and non-smooth.
The optimization problem is typically solved using (projected) gradient-based optimization algorithms.
In each step we set the patterns to non-negative values since light intensity cannot be negative.
We use the `L-BFGS` algorithm, which is a quasi-Newton method that is well-suited for non-smooth optimization problems.
Usually CUDA is used to accelerate the forward and backward propagation of the light patterns.

