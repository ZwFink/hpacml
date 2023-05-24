from collections import namedtuple
import subprocess
import os
import sys
import yaml
from yaml import CLoader

mandatoryOptions = ["build", "build_dir", "clean", "cmd", "exe", "input", "measure",  "quality_pattern", "root_dir", "analysis_dir", "profile_counters"]

class ApproxExecutorMissingFields(Exception):
    """ Application Descrption Fields Are Missing
    
    Attributes:
        Application Name -- aName
        Missing Fields --- fFields
    """

    def __init__(self, aName, fFields):
        self.aName = aName
        self.msg = "Missing Fields are:" + ",".join(fFields)
        super().__init__(self.msg)

    def __str__(self):
        return f'Application named \'{self.aName}\' is not well described\n\'{self.msg}\''

class ApproxExecutorCompilationError(Exception):
    """Compilation Error raised when compiling application

    Attributes:
        Application Name -- aName
        Application Compilation Output -- file name
    """

    def __init__(self, aName, stdout, stderr, msg="Compilation Error"):
        self.aName= aName
        self.stdout = stdout
        self.stderr = stderr
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f'Application named \'{self.aName}\' does not compile\n \'{self.stdout}\'\n\'{self.stderr}\''


class Executor:
    def __init__(self, name, configuration, data_base = None, copyBins = True):
        #name of the application
        self.name = name

        self.getMissingFields(configuration)

        self.copyBins  = copyBins
        self.configuration = configuration
        if copyBins:
            self.bin_dir = os.getcwd() + "/bin/%s/" % (name)
            os.makedirs( self.bin_dir, exist_ok = True)
        else:
            self.bin_dir = configuration["build_dir"]

        #command for compiling my application
        self.build_cmd = configuration["build"]
        #command for cleaning configuraiton
        self.clean_cmd = configuration["clean"]
        #Directory where building will take place
        self.build_dir = configuration["build_dir"]
        #Directory where I should store the data which needs to be processed by analysis
        self.analysis_dir = configuration["analysis_dir"]
        #Which counters should I monitor
        self.profile_counters = configuration["profile_counters"]
        #command for executing application 
        self.execute_cmd = self.createExecutionCommand(configuration)
        os.makedirs( self.analysis_dir, exist_ok = True)
        self.compiled = False
        if (data_base == None):
            self.data_base = "%s/%s.h5" % (self.analysis_dir, self.name)
        else:
            self.data_base = "%s/%s.h5" % (self.analysis_dir, data_base)

    def getDataBase(self):
        return self.data_base

    def createExecutionCommand(self, configuration):
        cmd = configuration["cmd"].replace("<input>", configuration["input"][0])
        cmd = cmd.replace("<output>", '%s')
        cmd = '%s/%s %s' % (self.build_dir,configuration["exe"], cmd)
        return cmd

    def build(self):
        try:
            p = subprocess.run( self.build_cmd, capture_output=True, cwd=self.build_dir,  shell=True )
        except Exception as e:
            print("Cmd: %s Failed" % self.build_cmd)
            sys.exit(0)

        if p.returncode != 0:
            raise ApproxExecutorCompilationError(self.name, p.stdout.decode('utf-8'), p.stderr.decode('utf-8'))
        self.compiled = True

    def prepare(self):
        if " copy" in self.configuration:
            for c in self.configuration["copy"]:
                if os.path.isdir(self.build_dir + "/" + c):
                    shutil.copytree( self.build_dir + '/' + copy, self.bin_dir + c)
                else:
                    shutil.copy( self.build_dir + '/' + c, self.bin_dir)

        if "soft_copy" in self.configuration:
            for c in self.configuration['soft_copy']:
                if not os.path.isfile( self.bin_dir+ c):
                    try:
                        os.symlink( self.build_dir + '/' + c , self.bin_dir + c)
                    except Exception as e:
                        print("Exception when creating soft link %s --> %s" %(self.build_dir + '/' + c , self.bin_dir + c))
                        sys.exit(0)

    def setEnvironment(self, extra_env):
        oldEnv = dict(os.environ)
        for v in extra_env:
            os.environ[v] = extra_env[v]
        return oldEnv

    def resetEnvironment(self, env):
        os.environ.clear()
        os.environ.update(env)
        pass

    def profileData(self, iterations=1):
        execution_conf = {}
        execution_conf["DATA_FILE"]=self.data_base
        execution_conf["EXECUTE_MODE"] = "DATA_PROFILE"
        oldEnv = self.setEnvironment(execution_conf)
        #currently execute binary 5 times to get performance counters
        #Data profile needs to happen only once
        self.execute(1)
        self.resetEnvironment(oldEnv)

    def profileCounters(self, iterations=5):
        execution_conf = {}
        execution_conf["DATA_FILE"]="%s/%s.h5" % (self.analysis_dir, self.name)
        execution_conf["EXECUTE_MODE"] = "TIME_PROFILE"
        execution_conf["PROFILE_EVENTS"] = ';'.join(self.profile_counters)
        oldEnv = self.setEnvironment(execution_conf)
        #currently execute binary 5 times to get performance counters
        self.execute(iterations)
        self.resetEnvironment(oldEnv)
        pass

    def execute(self, iterations):
        if (self.compiled == False):
            self.build()

        self.prepare()
        currentDir = os.getcwd()
        os.chdir(self.bin_dir)
        outDir = "./outputs"
        os.makedirs( outDir, exist_ok = True)
        for i in range(0,iterations):
            outName = outDir + "/file_%d.out" % (i)
            stdout = outDir + "/stdout.txt"
            stderr = outDir + "/stderr.txt"
            cmd = self.execute_cmd % (outName)
            try:
                p = subprocess.run( cmd, capture_output=True, shell=True )
                with open(stdout, "w") as f:
                    f.write(p.stdout.decode('utf-8'))

                with open(stderr, "w") as f:
                    f.write(p.stderr.decode('utf-8'))

            except Exception as e:
                print("Cmd: %s Failed" % cmd)
                sys.exit(-1)

        os.chdir(currentDir)

    def getMissingFields(self, config):
        missingOptions = []
        try:
            for f in mandatoryOptions:
                if f not in config:
                    missingOptions.append(f)
            if missingOptions:
                raise ApproxExecutorMissingFields(self.name, missingOptions)
        except(ApproxExecutorMissingFields):
            sys.exit(-1)
