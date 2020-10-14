#!/usr/bin/env node

const program = require(`commander`)
const dnn = require(`./platforms/dnn/index`)
const cnn = require(`./platforms/cnn/index`)
const configuration = require(`./platforms/config/index`)
 
program
    .version(`0.0.1`, `-v, --version`)

//This is for Scafolding a project
//DNN Nueral Network
program
    .usage(`[commands] <filename> `)
    .command(`dnn <name>`) 
    .option(`-g, --generate`, `Trigger Generation Operation`)
    .option(`-p, --project`)
    .action((name) => {
        dnn.nueralnet.dnn(name)
    })

//CNN Nueral Network
program
    .usage(`[commands] <filename> `)
    .command(`cnn <name>`) 
    .option(`-g, --generate`, `Trigger Generation Operation`)
    .option(`-p, --project`)
    .action((name) => {
        cnn.nueralnet.cnn(name)
    })
    
//CONFIGURATION
program
    .usage(`[commands] <config>`)
    .command(`config`)
    .action(()=>{
       configuration.config.config 
    })

//CONFIGURATION
program
    .usage(`[commands] <setup>`)
    .command(`setup`)
    .action(()=>{
       configuration.config.requirements
    })
program.parse(process.argv)