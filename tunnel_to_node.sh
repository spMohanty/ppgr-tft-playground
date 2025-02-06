#!/bin/bash


export MYPORT=18888
export HOST=kl004

ssh -L $MYPORT:localhost:$MYPORT $HOST