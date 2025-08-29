#!/bin/sh
killall ssh-agent; eval "$(ssh-agent)"
ssh-add -D
step ssh login sgroi.1852992@studenti.uniroma1.it --provisioner cineca-hpc
ssh tsgroi00@login02-ext.leonardo.cineca.it


