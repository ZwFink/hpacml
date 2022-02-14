import h5py
import os
import numpy as np

indexes = {
        "AVG" : 0,
        "STD" : 1,
        "MAX" : 2,
        "MIN" : 3
        }

class ApproxRegionNotFoundError(Exception):
    """Approximate Region Does not exist

    Attributes:
        region -- region name
        fName -- file name
    """

    def __init__(self, rName , fName , msg="Approx Region does not Exist"):
        self.rName=rName
        self.fName= fName
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f'region named \'{self.rName}\' does not exist inside file \'{self.fName}\''

class ApproxFileNotFoundError(Exception):
    """File Does not exist

    Attributes:
        file -- input file we try to read
        path -- where file was supposed to be stored
    """

    def __init__(self, mFile, path, msg="Approx Data File Does not Exist"):
        self.mFile = mFile
        self.path= path
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f'FILE \'{self.mFile}\' does not exist under path: \'{self.path}\''

class ApproxRegionIterator:
    def __init__(self, region, direction):
        self.direction = direction
        self.region = region
        self.indexes = []
        self.Values = None
        self.index = 0
        self.stop = 0
        if (direction == "input"):
            self.values = region.X()
            start = 0
            for _input in self.region.iInfo:
                end = start+ _input[0]
                self.indexes.append((start,end))
                start = end
            self.stop = len(self.indexes)
        elif (direction == "output"):
            self.values = region.Y()
            start = 0
            for _input in self.region.oInfo:
                end = start+ _input[0]
                self.indexes.append((start,end))
                start = end
            self.stop = len(self.indexes)
        else:
            print ("Should never happen")


    def __iter__(self):
        return self

    def __next__(self):
        while self.index < self.stop :
            (s,e) = self.indexes[self.index]
            self.index += 1
            return self.values[:,s:e]
        raise StopIteration

class approxRegion:
    def __init__(self, fName, rName, rInfo ,  **kwargs):
        self.fName = fName
        self.name = rName
        self.rInfo = rInfo

        self.iInfo = None
        self.oInfo = None

        if ( "ishape" in rInfo ):
            self.iInfo = np.array(rInfo.get("ishape"))
            self.iDim = self.iInfo[:,0].sum()
        if ( "oshape" in rInfo ):
            self.oInfo = np.array(rInfo.get("oshape"))
            self.oDim = self.oInfo[:,0].sum()

        self._repr = "Region Name: %s |\t Num Inputs : %s |\t Num Ouputs : %s |\t Profile Information : %s |"
# Data will be read as late as possible
        self.data = None
        self.prefetched = False
        self.x_values = None
        self.y_values = None
        self.profile = None
        self.prediction_code = ""
        self.profile_data = {}


    def getDBPath(self):
        return self.fName

    def __repr__(self):
      if isinstance(self.oInfo, type(None)) and isinstance(self.iInfo, type(None)):
        return  self._repr % (self.name, 'Empty', 'Empty', self.hasProfileData())
      elif isinstance(self.oInfo, type(None)):
        return  self._repr % (self.name, str(self.iInfo[:,0].sum()), 'Empty', self.hasProfileData())
      elif isinstance(self.iInfo, type(None)):
        return  self._repr % (self.name, 'Empty', str(self.oInfo[:,0].sum()), self.hasProfileData())
      else:
        return  self._repr % (self.name, str(self.iInfo[:,0].sum()), str(self.oInfo[:,0].sum()), self.hasProfileData())

    def hasProfileData(self):
        if ("ProfileData" not in  self.rInfo ):
            print("Application does not have available profile data")
            return False
        return True

    def getProfileEvents(self):
        if (not self.hasProfileData()):
            print("Application does not have profiling info")
            sys.exit(0)
        profile_data = self.rInfo["ProfileData"]
        keys = profile_data.keys()
        return list(keys)

    def getProfileData(self):
        if (not self.hasProfileData()):
            print("Application does not have profiling info")
            sys.exit(0)

        profile_data = self.rInfo["ProfileData"]
        keys = profile_data.keys()
        data = {}
        for k in keys:
            data[k] = np.array(profile_data.get(k))[:,indexes["AVG"]]
        return data

    def getProfileTime(self, counter ):

        if ( counter in self.profile_data ):
            return self.profile_data[counter]

        if ("ProfileData" not in  self.rInfo ):
            print("Application does not have available profile data")
            return None

        profile_group = self.rInfo["ProfileData"]

        if (counter not in profile_group.keys()):
            print("This profiling data do not exist, available counters are:")
            print(", ".join(v for v in profile_group.keys()))
            return None

        temp = np.array(profile_group.get(counter))
        self.profile_data[counter] = np.average(temp[:,0])
        return self.profile_data[counter]

    def getX(self):
        if self.iInfo is None:
            return None
        if ( not self.prefetched ):
            self.prefetched = True
            self.data = np.array(self.rInfo.get("data"))
        endX = self.iInfo[:,0].sum()
        self.x_values = self.data[:, 0:endX]
        return self.x_values

    def getY(self):
        if self.oInfo is None:
            return None

        if ( not self.prefetched ):
            self.prefetched = True
            self.data = np.array(self.rInfo.get("data"))
        startY = self.iInfo[:,0].sum()
        self.y_values = self.data[:, startY:]
        return self.y_values

    def X(self):
        return self.getX()

    def Y(self):
        return self.getY()

    def getRegionName(self):
        return self.name

    def inputs(self):
        return ApproxRegionIterator(self,"input")

    def outputs(self):
        return ApproxRegionIterator(self,"output")

    def getNumInputs(self):
        return self.iDim

    def getNumOutputs(self):
        return self.oDim


class ApproxApplicationIterator:
    def __init__(self, application):
        self.application = application
        self.keys = application.getRegionNames()
        self.index = 0

    def __next__(self):
        if self.index < len(self.keys):
            cName = self.keys[self.index]
            self.index += 1
            return self.application[cName]
        raise StopIteration


class approxApplication:
    def __init__ (self, path_to_file,  **kwargs):
        path = '/'.join(path_to_file.split("/")[:-1])
        application =path_to_file.split("/")[-1]
        application = application.split(".")[0]
        self.application = application
        self.dPath = path
        self.fName = path_to_file
        self.AppHasOutput = False
        self.AppHasInput = False
        self.iApp = None
        self.oApp = None

        if (not os.path.exists(self.fName)):
            raise ApproxFileNotFoundError(application, self.dPath)

        self.dFile = h5py.File(self.fName, "r")
        fileKeys = list(self.dFile.keys())
        if 'ApplicationOutput' in fileKeys:
          self.AppHasOutput = True
          fileKeys.remove('ApplicationOutput')
        if 'ApplicationInput' in fileKeys:
          self.AppHasInput = True
          fileKeys.remove('ApplicationInput')
        self.rKeys = fileKeys
        self.regions = {}

    def makeRegion(self, rName):
        if rName not in self.regions:
            self.regions[rName] = approxRegion(self.fName, rName, self.dFile[rName])

    def __len__(self):
        return len(self.rKeys)

    def getApplicationInput(self):
        if not self.AppHasInput:
            return None

        if self.iApp is None:
            self.iApp = {}
            tmp = self.dFile['ApplicationInput'].keys()
            for t in tmp:
              self.iApp[t] = np.array(self.dFile['ApplicationInput'][t])

        return self.iApp

    def getApplicationOutput(self):
        if not self.AppHasOutput:
            return None

        if self.oApp is None:
            self.oApp = {}
            tmp = self.dFile['ApplicationOutput'].keys()
            for t in tmp:
              self.oApp[t] = np.array(self.dFile['ApplicationOutput'][t])

        return self.oApp

    def __getitem__(self,rName):
        if rName == 'ApplicationOutput':
          self.getApplicationOutput()
        if rName == 'ApplicationInput':
          self.getApplicationInput()

        if rName not in self.rKeys:
            raise ApproxRegionNotFoundError(rName,self.fName)
        self.makeRegion(rName)
        return self.regions[rName]

    def __repr__(self):
        return "%s\n\t\t -> App:%s\n\t\t -> Path:%s" % (self.__class__.__name__, self.application, self.dPath)

    def __iter__(self):
        return ApproxApplicationIterator(self)

    def getRegionNames(self):
        return list(self.rKeys)

    def xValues(self,rName):
        if rName not in self.rKeys:
            raise ApproxRegionNotFoundError(rName,self.fName)
        self.makeRegion(rName)
        return self.regions[rName].getX()

    def yValues(self,rName):
        if rName not in self.rKeys:
            raise ApproxRegionNotFoundError()
        makeRegion(rName)
        return self.regions[rName].getY()

