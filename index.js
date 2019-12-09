#!/usr/bin/env node

const program = require(`commander`)
const dnn = require(`./platforms/dnn/index`)
const configuration = require(`./platforms/config/index`)
 
program
    .version(`0.0.0`, `-v, --version`)

//This is for Scafolding a project
//DNN Nueral Network
program
    .usage(`[commands] <filename> <type> <options1> <options2>`)
    .command(`dnn <name> <type>`) 
    .option(`-g, --generate`, `Trigger Generation Operation`)
    .option(`-p, --project`)
    .action((name) => {
        dnn.nueralnet.dnn(name)
    })


//CONFIGURATION
program
    .usage(`[commands] <config> <hostpc-name>`)
    .command(`config <hostname>`)
    .action((hostname)=>{
        configuration.configuration.config(hostname)
    })


program.parse(process.argv)