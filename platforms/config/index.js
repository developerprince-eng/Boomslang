/*
PROJECT TITLE           : BoomSlangCLI
PROJECT ALIAS           : fluid-serpent
PROJECT OWNER           : DeveloperPrince
PROJECT LEAD            : DeveloperPrince (Prince Kudzai Maposa)
AFFILIATION             : Semina
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

let config = () => {
    let winenv = () => {

    }

    let unix = () => {
        cp.execSync(`cd /tmp`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`SWITCH to Temporary Folder`)
            console.log(stderr)
        })

        cp.execSync(`curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Download Anaconda to Temporary Folder`)
            console.log(stderr)
        })

        cp.execSync(`bash Anaconda3-2019.03-Linux-x86_64.sh`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Install Anaconda`)
            console.log(stderr)
        })

        cp.exec(`bash Anaconda3-2019.03-Linux-x86_64.sh`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Install Anaconda`)
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

let requirements = () => {
    let winenv = () => {

    }

    let unix = () => {
        cp.execSync(`conda create -n aienv python=3.6`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`CREATE Conda Environment`)
            console.log(stderr)
        })

        cp.execSync(`conda activate aienv`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Download Anaconda to Temporary Folder`)
            console.log(stderr)
        })

        cp.execSync(`pip install -r /usr/lib/node_modules/boomslang-cli/templates/requirements.txt`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Install Dependencies`)
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

exports.config = {config, requirements}