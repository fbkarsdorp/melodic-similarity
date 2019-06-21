#!/bin/sh

g++ -I /usr/local/include/libmusical/ -c ismir2019/*.cpp
g++ -o alignismir2019 apps/alignismir2019.cpp *.o -lmusical -L /usr/local/lib/ -I /usr/local/include/libmusical/ -I ismir2019
g++ -o ismir2019distmat apps/ismir2019distmat.cpp *.o -lmusical -L /usr/local/lib/ -I /usr/local/include/libmusical/ -I ismir2019
rm *.o
