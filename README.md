# Physical information neural network for nonlinear integro-differential equations with degenerate kernel scheme
This repository hosts the source code for the paper "Physical information neural network for nonlinear integro-differential equations with degenerate kernel scheme" aiming at the development of a novel Artificial lntelligence (A)-based This repository hosts the source code for the paper "Physical information neural network for nonlinear integro-differential equations with degenerate kernel scheme" aiming at the development of a novel Artificial lntelligence (A)-based Physics-lnformed neural network for efficient physical modeling and computation with limited data. The novel Alframework enriches mashine learning with physical science for accurate predictions even with few noisymeasurements from an experiment. neural network for efficient physical modeling and computation with limited data. The novel Alframework enriches mashine learning with physical science for accurate predictions even with few noisymeasurements from an experiment.
# Table of Contents
## Introduction
## Contact
## Citation
# Introduction
In recent years, the Physical Information Neural Network (PINN) model has become an interdisciplinary research hotspot in nonlinear science and artificial intelligence. Nonlinear integro-differential equations (IDEs), as an essential mathematical model in science and engineering, often face challenges in forward problem analysis and inverse problem solving due to the complexity of their kernel functions. This paper proposed a PINN framework that combines degenerate kernel schemes to solve the IDEs of mathematical models in nonlinear science including forward problems and inverse problems. Herein, the general smooth continuous IDEs are first approximated to the IDEs with only a degenerate kernel, and the equivalent nonlinear differential equations are obtained by introducing the auxiliary differential operators with new boundary conditions to replace the integral ones. For the function to be solved in the original IDEs and the new functions in the auxiliary differential operators, the independent full connection deep neural networks (FCDNN) are established. By constructing the loss function based on the equivalent nonlinear differential equations and all boundary conditions and initial conditions, this PINN is trained to realize the solution of the nonlinear IDEs forward problem by using the Adam optimizer. By constructing new loss components based on the measured values of the function to be solved in the forward problem, the IDEs inverse problems further can be solved by PINN, such as unknown parameters or source items in IDEs. Detailed numerical analysis of the forward problem, inverse problem, and some high-dimensional problems of the nonlinear IDEs shows that the proposed PINN has high accuracy and universality for such nonlinear problems. In addition, the effects of the characteristics of the solution, the network framework, the forms of activation function and loss function and physical information distribution points on the convergence of the PINN method are discussed in detail. The universal PINN solution for nonlinear IDEs can serve the IDEs application in science and engineering.
