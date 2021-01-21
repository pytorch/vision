#!/bin/bash

match=`git grep -A 1 Parameters | grep "\-\-\-"`
if [ ! -z "$match" ]
then
  echo "Bad docstring format. Please use the Args: syntax to specify arguments"
  exit 1
fi