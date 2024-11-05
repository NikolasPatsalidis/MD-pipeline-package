from __future__ import print_function
import numpy as np
import re

class FileLineReader:
    def __init__(self, filename, commentStart='#'):
        self.lineno = 0
        self.commentStart = commentStart
        try:
            self.file = open(filename,'r')
        except:
            self.file = None
            raise

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def read(self):
        if self.file is None:
            raise Exception("No file to read from")
        while True:
            line = self.file.readline()
            if len(line) == 0:
                return line
            self.lineno += 1
            line = line.strip()
            if len(line) == 0:
                continue
            if self.commentStart is not None:
                pos = line.find(self.commentStart)
                if pos >= 0:
                    line = line[0:pos]
                line = line.strip()
                if len(line) == 0:
                    continue
            return line

class FloatCol:
    def __init__(self, colName, nRows):
        self.name = colName
        self.data = np.zeros((nRows,), np.float64)
        self.position = 0

    def appendString(self, s):
        self.data[self.position] = float(s)
        self.position += 1

class IntCol:
    def __init__(self, colName, nRows):
        self.name = colName
        self.data = np.zeros((nRows,), np.int64)
        self.position = 0

    def appendString(self, s):
        self.data[self.position] = int(s)
        self.position += 1

class StrCol:
    def __init__(self, colName, nRows):
        self.name = colName
        self.data = np.zeros((nRows,), dtype='S200')
        self.position = 0

    def appendString(self, s):
        self.data[self.position] = s
        self.position += 1

def isLammpsDataSorted(data):
    if not 'id' in data:
        return False
    ids = data['id']
    for i in range(len(ids)):
        if ids[i] != i+1:
            return False
    return True        

def sortLammpsData(data):
    if not 'id' in data:
        return False
    ids = data['id']
    to_sort = []
    i = 0
    for id in ids:
        to_sort.append((id,i))
        i += 1
    to_sort.sort()
    for k in data['cols']:
        column_data = data[k]
        new_data = np.zeros((len(column_data),), column_data.dtype)
        for i in range(len(to_sort)):
            new_data[i] = column_data[to_sort[i][1]]
        data[k] = new_data
    return True
      

class LammpsTrajReader:
    def __init__(self, filename):
        self.reader = None
        self.reader = FileLineReader(filename)
        self.step_re = re.compile('^\s*ITEM\s*:\s*TIMESTEP\s*$', re.IGNORECASE)
        self.nats_re = re.compile('^\s*ITEM\s*:\s*NUMBER\s*OF\s*ATOMS\s*$', re.IGNORECASE)
        self.box_bounds_re = re.compile('^\s*ITEM\s*:\s*BOX\s*BOUNDS\s+.*$', re.IGNORECASE)
        self.atoms_re = re.compile('^\s*ITEM\s*:\s*ATOMS\s+(.*)$', re.IGNORECASE)
        self.measure_box_bounds_re = re.compile('^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+(?:\.?[0-9]+)?)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+(?:\.?[0-9]+)?)?)(\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+(?:\.?[0-9]+)?)?))?\s*$', re.IGNORECASE)
#                                                     ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+(?:\.?[0-9]+)?)?)   
        self.box_bounds = np.zeros((3,3))
        self.strict_atom_numbering = True
    
    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.reader.file.close()
        
    def __del__(self):
        if self.reader is not None:
            del self.reader

    def dataTypeOfField(self, fieldName):
        floats = {'xu','yu','zu','x','y','z','vy','vy','vz','fz','fy','fz','mass'}
        ints = {'id','mol','type'}
        # strs = set('type')
        if fieldName in floats:
            return np.float64
        elif fieldName in ints:
            return np.int64
        else:
            return np.string_

    def isSorted(self):
        return isLammpsDataSorted(self.data)

    def sort(self):
        return sortLammpsData(self.data)

    def readNextStep(self):
        step_no = None
        while step_no is None:
            line = self.reader.read()
            if len(line) == 0:
                break
            if self.step_re.match(line) is not None:
                step_no = int(self.reader.read().strip())
                break
        if step_no is None:
            return None
        #print "Step", step_no

        nats = None
        while nats is None:
            line = self.reader.read()
            if len(line) == 0:
                break
            if self.nats_re.match(line) is not None:
                nats = int(self.reader.read().strip())
                break
        if nats is None:
            return None
        #print "number of atoms", nats

        got_box_bounds = False
        while not got_box_bounds:
            line = self.reader.read()
            if len(line) == 0:
                break
            if self.box_bounds_re.match(line) is not None:
                line = self.reader.read()
                parts = line.split()
                self.box_bounds[0,0] = float(parts[0])
                self.box_bounds[0,1] = float(parts[1])
                self.box_bounds[0,2] = 0.0
                if len(parts) > 2:
                    self.box_bounds[0,2] = float(parts[2])
                line = self.reader.read()
                parts = line.split()
                self.box_bounds[1,0] = float(parts[0])
                self.box_bounds[1,1] = float(parts[1])
                self.box_bounds[1,2] = 0.0
                if len(parts) > 2:
                    self.box_bounds[1,2] = float(parts[2])
                line = self.reader.read()
                parts = line.split()
                self.box_bounds[2,0] = float(parts[0])
                self.box_bounds[2,1] = float(parts[1])
                self.box_bounds[2,2] = 0.0
                if len(parts) > 2:
                    self.box_bounds[2,2] = float(parts[2])
                # if len(line) == 0:
                #     break
                got_box_bounds = True
                break
        if not got_box_bounds:
            return None
        #print "read box bounds"

        atom_cols = None
        while atom_cols is None:
            line = self.reader.read()
            if len(line) == 0:
                break
            m = self.atoms_re.match(line)
            if m is not None:
                atom_cols = m.group(1).split()
                break
        if atom_cols is None:
            return None
        # print("atom colums", atom_cols)

        data = []

        for colName in atom_cols:
            dataType = self.dataTypeOfField(colName)
            if dataType == np.float64:
                data.append(FloatCol(colName, nats))
            elif dataType == np.int64:
                data.append(IntCol(colName, nats))
            elif dataType == np.string_:
                data.append(StrCol(colName, nats))
            else:
                raise Exception('Unexpected data type.')
            
        for i in range(nats):
            cols = self.reader.read().split()

            for datai in range(len(data)):
                data[datai].appendString(cols[datai])

            # if ai is not None:
            #     if self.strict_atom_numbering and int(cols[ai]) != i+1:
            #         raise Exception('Invalid atom index, atoms should be sorted.')
            #     atIds[i] = int(cols[ai])


        ret = {'step_no': step_no,
                'box_bounds': self.box_bounds.copy(),
                'cols':[]}
        for col in data:
            ret['cols'].append(col.name)
            ret[col.name] = col.data
        self.data = ret
        return ret
