#!/bin/sh

make clean
premake4 gmake
make -j 8
