/*
PROJECT TITLE           : BoomSlangCLI
PROJECT ALIAS           : fluid-serpent
PROJECT OWNER           : DeveloperPrince
PROJECT LEAD            : DeveloperPrince (Prince Kudzai Maposa)
AFFILIATION             : Semina
PROJECT CONTRIBUTORS    : Tinashe Mabika
PROJECT BRIEF           : This is command line interface which will enable developers to scaffold 
                          project of any nature which are ai based, currently focusing 
                          python.

Happy Coding 
*/
/*--------------Modules-------------------*/
const cp = require(`child_process`)
const fs = require(`fs`)
const isWin = process.platform === "win32"
const internetAvailable = require(`internet-available`)

let cnn = (name) => {

    let winenv = () => {
        cp.execSync(`mkdir ${name} `, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`CREATED FOLDER:  ${name} & Switched to Root Project Folder`)
            console.log(stderr)
        })
       
        cp.execSync(`cp ~/usr/local/lib/boomslang/templates/cnn/. /$name`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Copied Core Files to  ${name} Folder`)
            console.log(stderr)
        })
    }

    let unixenv = () =>{
        cp.execSync(`mkdir ${name} `, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`CREATED FOLDER:  ${name} `)
            console.log(stderr)
        })

        cp.execSync(`cd ${name}`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Switched to Root Project Folder`)
            console.log(stderr)
        })
       
        cp.execSync(`cp -r /usr/lib/node_modules/boomslang-cli/templates/dnn/ ${name}`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Copied Core Files to  ${name} Folder`)
            console.log(stderr)
        })
    }
    //CHECK FOR SYSTEM ENVIRONMENT
    if(isWin){
        winenv()
    }
    else{
        unixenv()
    }
}

exports.nueralnet = {cnn}