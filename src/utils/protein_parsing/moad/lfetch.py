# MIT License

# Copyright (c) 2021 Luca Gagliardi
# Affiliation Istituto Italiano di tecnologia


# MOAD database:
# "Since all structures in
# Binding MOAD must contain a valid ligand, the likelihood of an
# invalid ligand occupying a biologically relevant site is greatly
# reduced. While it is still possible, the rate of such occurrence is
# much less than using all the structures in the Protein Data Bank"
# https://github.com/lucagl/MOAD_ligandFinder


import os

import glob
import re
import shutil
import subprocess
import sys
from datetime import timedelta
from sys import exit
from time import localtime, sleep, strftime, time

import numpy as np
import requests
from requests.exceptions import HTTPError

URL = "http://bindingmoad.org/pdbrecords/index/"

# Rest of the code...

#### USEFUL FUNCTIONS
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


###### USEFUL CLASSES
# * Time profiling class


class ContinueI(Exception):
    pass


def secondsToStr(elapsed=0):
    if elapsed == 0:
        return strftime("LOCAL TIME = %Y-%m-%d %H:%M:%S", localtime())
    else:
        return "ELAPSED TIME: " + str(timedelta(seconds=elapsed))


class Crono(object):
    def init(self):
        self._elapsed = 0
        self._start = time()
        return secondsToStr(self._elapsed)

    def get(self):
        end = time()
        self._elapsed = end - self._start
        return secondsToStr(self._elapsed)


#########
# * Error handler class

_warning = "<WARNING> "
_break = "<ERROR> "


class Error(object):
    n_errors = 0

    def __init__(self):
        self.value = 0
        self.info = " "
        self._status = None

    def reset(self):
        self.__init__()

    def partial_reset(self):
        self.info = ""
        self._status = None

    def put_value(self, n):
        # safe writing avoids to overwrite major errors
        if (self.value != 2) and (self.value != 3):
            self.value = n
        return

    def build_status(self):
        if self.value == 1:
            self._status = _warning
        elif self.value == 2:
            self._status = _break
        elif self.value == 3:
            self._status = _warning
        else:
            pass
        return

    def get_info(self):
        self.build_status()
        # print(self._status + self.info)
        return self._status + self.info

    def write(self, errFile):
        try:
            errFile.write("\n " + self.get_info())
        except OSError:
            raise NotImplementedError("Error log file not found..")
        return

    def handle(self, errFile):
        # OBS: Only when errors are handled the static counter is updated
        if self.value == 0:  # do nothing
            # self.reset()
            return
        if self.info == "":
            return
        Error.n_errors += 1
        try:
            errFile.write(self.get_info() + "\n ")
        except OSError:
            raise NotImplementedError("Error log file not found..")
        if self.value == 4:
            errFile.close()
            try:
                raise Exception("Exiting for major error")
            except Exception:
                print("Exiting for major error")
                exit()
        self.partial_reset()
        return


#######################################
#####


########## Query function for MOAD #########
# /pdbrecords/exclusion/4/1hel
def queryMOAD(pdb_name, patience):
    """
    Given a pdb name, returns the number of valid ligands with a map than can be used (from extractLigand())
    to extract the ligand coordinates from the pdb file, distinguishing between separate ligands.
    To do so the functions fetch data from the MOAD database.
    """
    err = Error()
    i = 0
    while 1:
        try:
            response = requests.get(URL + pdb_name, allow_redirects=True)
            # print(response.url)
            response.raise_for_status()
            break
        except HTTPError:
            if i > 3 or (not patience):  # try 3 times this
                print("HTTP error occurred: No match in the MOAD database for " + pdb_name)
                err.value = 2
                err.info = "HTTP error occurred. No match in the MOAD database for " + pdb_name
                return err, []
            print("\r\t\tHTTP error: Attempt  " + str(i), end="")
        except Exception:
            if i >= 300 or (not patience):  # 5 mins of attempts
                print(
                    "Other error occurred for "
                    + pdb_name
                    + " (probably internet connection issues..)"
                )
                err.value = 2
                err.info = (
                    "Other error occurred for "
                    + pdb_name
                    + " (probably internet connection issues..)"
                )
                return err, []
            print("\r\t\tConnection error: Attempt  " + str(i), end="")
        sleep(1)
        i += 1

    # OBS also from response url I could deduce no match..
    raw = response.text
    pageTxt = [x.strip() for x in raw.split("\n")]

    # Verify entry exists in MOAD
    moadError = "\t"
    gotMatch = False
    ligNames = set()  # set is like a dict with only keys (duplicates are not considered)
    for c, line in enumerate(pageTxt):
        match = re.match('(<p class="sidebar-title">Showing PDB:\s*)([^</p>]*)', line)
        match2 = re.match("([^.]+\s*<br/>)", line)
        matchLig = re.match("(^Ligand no:\s*\d+)([;?\s]*Ligand:\s*)([^;]*)", line)
        if match2:
            # moadError= "\t" + match2.group() + "\t" + pageTxt[c+1]
            moadError = (
                "The pdb "
                + pdb_name
                + " was excluded from MOAD for the following reason:  "
                + pageTxt[c + 1]
            )
        if match:
            gotMatch = True
        if matchLig:
            # print(matchLig.group(2))
            ligNames.add((matchLig.group(3)))
            # got_name = match.group(2)
            # print(got_name)
            # if(got_name == pdb_name):
            #     print(match.group())
            #     print("OK")
            #     break
    if not gotMatch:
        err.put_value(2)
        err.info = moadError
        # print(response.history)
        # print("Ligand or entry excluded from MOAD. Check error log")
        return err, []

    # Here on I'm sure the page has been correctly loaded
    # Looking for ligands

    # print(len(ligNames),ligNames)

    if not ligNames:
        err.put_value(2)
        err.info = "Unkonown error. Unable to find ligands in MOAD for " + pdb_name
        return err, []
    firstMatch = np.ones(len(ligNames), bool) * False

    ligands = []
    s = 0
    for l in ligNames:
        # print(l)
        for c, line in enumerate(pageTxt):
            matchname = re.match("<td>" + l + "\s*</td>", line)
            if matchname and (not firstMatch[s]):
                firstMatch[s] = True
                # print(line)
                nextline = pageTxt[c + 1]
                # matchChainRest = re.findall("([A-Z]:[\d]+)",nextline)
                # very rarely chain is a number!
                matchChainRest = re.findall("([\w0-9]:[\d]+)", nextline)

                # print(nextline)
                # print(matchChainRest)

                # print("Number of distint ligands for current name:",len(matchChainRest))
                namelist = l.split()
                for cr in matchChainRest:
                    # this will be used as filename containing ligand coordinates
                    # set() to remove duplicates

                    name = repr(namelist).replace("[", "").replace("]", "").replace("'", "")

                    filename = name.replace(", ", "_") + "_" + cr.replace(":", "")

                    chain, number = re.match("([\w0-9]):([\d]+)", cr).groups()

                    resid = [str(i) for i in range(int(number), int(number) + len(namelist))]

                    ligands.append(
                        {
                            "filename": filename,
                            "name(s)": namelist,
                            "chain": chain,
                            "resid(s)": resid,
                        }
                    )

        s += 1

    # <fieldset id="fieldset_ligandinfo" class="coolfieldset">
    # <tbody>
    #                             <tr>
    #         <td>NAG NAG NAG </td>
    #         <td>B:1;<br> </td>
    #         <td>Valid;<br> </td>

    n_ligands = len(ligands)
    if n_ligands == 0:
        err.put_value(2)
        err.info = "Unkonown error. Unable to find ligands in MOAD for " + pdb_name
        return err, []
    # print("TOTAL NUMBER VALID DISTINT LIGANDS OF THE QUERY= %d" %n_ligands)
    return err, ligands


def extractLigand(pdbName, dictList, savepath, onlyXYZ, extractPQR, purgePDB):
    """
    Given the map produced by queryMOAD() and the pdb file, creates xyz files of the heavy atoms of the ligand (*)
    Also produces a temp file containing the pdb without the ligand coordinates EXTRACTED neither HETATM.
    Since there could be some difficulties in extracting ligand coordinates due to mismatch with the MOAD database,
    there are no guarantees the structure is completely purged of ligands before conversion to PQR if ligands expected naming scheme not obserrved.
    However this scenario is rare and a error could prevent the conversion.
    In any case all labelled HETATM are correclty elided before conversion to PQR.

    Any manipulation of the standard file or re-interpretation of the MOAD search pattern is signaled. (examples: 4mdr (only for 1 of the 2 ligands),4mdr )

    *) Heavy atoms are kept recognizing Hydrogens in the ligand (example pdb_code: 1cka)

    NOTE: in purged PDB extraction, if PQR conversion is performed, the line containing SEQADV is also removed
    since this very rarely was creating problems in conversion.
    """

    import copy

    key_dictionary = []
    for li in dictList:
        # print(li)
        d2 = copy.deepcopy(li)
        key_dictionary.append(d2)

    import re

    err = Error()
    matchPerLigand = np.zeros(
        len(key_dictionary)
    )  # 0: not match, 1: full match (how expected from MOAD), -1: partial match

    comment = ["#", "REMARK", "MASTER", "END"]
    info = ["SOURCE", "COMPND", "TITLE", "HEADER", "JRNL", "AUTHOR", "REVDAT", "KEYWDS", "EXPDTA"]
    crist = ["CRYST1", "ORIGX[0-9]", "SCALE[0-9]", "MTRIX[0-9]", "TVECT"]
    if extractPQR:
        primStruct = [
            "DBREF",
            "SEQRES",
            "MODRES",
            "SITE",
        ]  #'SEQADV' skipping, sometimes creates problem to keep this line in pd2prq..
    else:
        primStruct = ["DBREF", "SEQRES", "MODRES", "SITE", "SEQADV"]
    connectivity = ["CONECT", "SSBOND", "LINK", "HYDBND", "SLTBRG", "CISPEP"]
    secStruct = ["HELIX", "SHEET", "TURN"]
    protein = ["ATOM", "MODEL", "SIGATM", "ANISOU", "SIGUIJ", "TER", "ENDMDL"]
    het = ["HET\s+", "HETNAM", "HETSYN", "FORMUL"]  # HET infos (no coordinates)

    # proteinExtract = ['CRYST1','ATOM','END']

    keep = comment + info + crist + primStruct + secStruct + connectivity + protein + het
    # skip = proteinExtract
    keep = "(?:^% s)" % "|^".join(keep)  # match beginning

    if pdbName.endswith(".pdb"):
        pdbName = pdbName[:-4]

    try:
        pdb_file = open(pdbName + ".pdb", "r")
    except Exception:
        err.info = "Did not find pdb file for " + pdbName
        err.put_value(2)
        return err, matchPerLigand

    if os.stat(pdbName + ".pdb").st_size == 0:
        err.info = "Empty file for  " + pdbName
        err.put_value(2)
        return err, matchPerLigand

    pdb_data = pdb_file.readlines()

    savepath = savepath + "/"

    ntrial = []
    initialLength = []
    for s in range(len(key_dictionary)):
        initialLength.append(len(key_dictionary[s]["name(s)"]))
        ntrial.append(0)

    ################### MAIN LOOP FETCHING LIGAND COORDS AND PREPARING TMP FILE FOR PDB2PQR #####################
    ligandFile = []

    if onlyXYZ:
        extension = ".xyz"
    else:
        extension = ".pdb"
    for s, i in enumerate(key_dictionary):
        duplicates = ""
        d = 0
        while 1:
            lfname = savepath + i["filename"] + duplicates + extension
            if os.path.isfile(lfname):
                duplicates = "_" + str(d + 1)
                # Modify map externally
                dictList[s]["filename"] = i["filename"] + duplicates
            else:
                break
            d += 1
        ligandFile.append(lfname)

    while 1:
        fo = [open(lf, "w") for lf in ligandFile]
        matches = []
        # This tries to not search in order for some resname, since it can happen MOAD entries are more than those actually in the PDB
        # Also we relax on having a partial match. This because sometimes some of the last resnames and ids are missing.
        for s in range(len(key_dictionary)):
            if ntrial[s] > 0:
                del key_dictionary[s]["name(s)"][0]

        # print(key_dictionary)

        # ******** Build a dictionary of the matching pattern searched ********
        for d in key_dictionary:
            names = d["name(s)"]  # sublist of identifiers belonging to same ligand
            chain = d["chain"]  # single letter per ligand
            ids = d["resid(s)"]  # sublist as long as names
            m = []
            for i in range(len(names)):
                name = names[i]
                resid = ids[i]
                # (?:) non capturing group
                m.append(
                    "(?:ATOM|HETATM)\s*[\d]+\s+[\S]+\s*[A-Z]*\s*"
                    + name
                    + "\s+"
                    + chain
                    + "\s*"
                    + resid
                    + "[A-Z]*\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s+.*([A-Z])\s*$"
                )
            # NOT matching H at the end: #m.append("(?:ATOM|HETATM)\s+[\d]+\s+[\S]+\s*[A-Z]*\s*"+name+"\s+"+chain+"\s*"+resid+"\s+(\-*\d*\.?\d+)\s+(\-*\d*\.?\d+)\s+(\-*\d*\.?\d+)(?!.*\\bH\\b).*$")
            # m.append("HETATM\s+[\d]+\s+[\S]+\s*[A-Z]*\s*"+name+"\s+"+chain+"\s*"+resid+"\s+(\-*\d*\.?\d+)\s+(\-*\d*\.?\d+)\s+(\-*\d*\.?\d+)(?!.*\\bH\\b).*$")
            #                            ^letter, number and some special characters                                         ^Do not take H atoms
            matches.append(m)
        ##########
        #######
        make_purgedPDB = False
        if extractPQR or purgePDB:
            make_purgedPDB = True
            if purgePDB:
                tmpFile = open(
                    pdbName + "_clean.pdb", "w"
                )  # contains original files without FOUND ligand and hetatm
            else:
                tmpFile = open("tmp", "w")
        # print(matches)
        ligandAtoms = 0
        # print(matches)
        matchPerResName = [np.zeros(len(l)) for l in matches]  # mlist is whithin a single ligand

        for line in pdb_data:
            ligandMatched = False
            HEAVY = True
            for s, mlist in enumerate(matches):
                if not HEAVY:
                    break
                for k, m in enumerate(mlist):
                    match = re.match(m, line)
                    if match:
                        matchPerResName[s][k] = 1
                        # print (match.groups())
                        if (match.groups()[-1] == "H") or (match.groups()[-1] == "D"):
                            HEAVY = False
                            break
                        ligandAtoms += 1
                        # print(line)
                        # print(match.groups())
                        coordline = tuple(
                            [it for it in map(lambda x: float(x), match.groups()[0:3])]
                        )
                        # print(coordline)
                        if onlyXYZ:
                            fo[s].write("%f\t%f\t%f\n" % coordline)
                        else:
                            fo[s].write(line)
                        ligandMatched = True
                        break
            if not HEAVY:
                # Here only if ligand line is matched
                # skipping also the line for the tmp file-->Avoid spurious light ligand atom in purged pdb..
                continue
            if not ligandMatched and make_purgedPDB:
                # write wathever is not ligand nor HETATM nor not heavy
                if re.match(keep, line):
                    # print(line)
                    tmpFile.write(line)  # write all lines starting with "keep" to tmp file
                    continue
        if make_purgedPDB:
            tmpFile.close()
        for f in fo:
            f.close()
        #######
        # Here on matchPerResName contains a map of all RES names matched per ligand. Where the first index runs over the ligands
        # print("matchArray:",matchPerResName)
        # print("trials array:",ntrial)
        ok = 0
        fail = 0
        for s, l in enumerate(matchPerResName):
            # print(l)
            # print(str(initialLength[s]) +" VS " +str(np.sum(l)))
            # s=ligand index
            # Tollerance over partial matches + trying to correct for bad alignement in the MOAD database
            # We use initial lenght because if a match isot satisfied with less than half, is not accepted since the discrepancy with MOAD is too big
            if (len(key_dictionary[s]["name(s)"]) == 0) or (len(l) < initialLength[s] // 2):
                fail += 1
                ntrial[s] = -1
            elif (np.sum(l) < initialLength[s] // 2) and (len(key_dictionary[s]["name(s)"]) > 1):
                ntrial[s] += 1
            else:
                if np.sum(l) == initialLength[s]:
                    matchPerLigand[s] = 1
                else:
                    # partial match
                    err.put_value(1)
                    matchPerLigand[s] = -1
                    # ligandInfo = repr(key_dictionary[s]["name(s)"]).replace("[","").replace("]","").replace("'","").replace(", ","_")+" chain="+key_dictionary[s]["chain"]
                    ligandInfo = key_dictionary[s]["filename"]
                    err.info += (
                        "\n Corrected "
                        + ligandInfo
                        + "  of PDB= "
                        + pdbName
                        + ": Needed to skip first "
                        + str(ntrial[s] + 1)
                        + " residue names out of "
                        + str(initialLength[s])
                    )
                ok += 1
        # print("**OKS:", ok)
        # print("**FAILS:", fail)
        ### END CONDITION ##
        if ok + fail == len(matchPerResName):
            if ok == 0:
                err.info = "**Did not suceed fetching ligands for " + pdbName
                err.put_value(2)
                for s in range(len(key_dictionary)):
                    subprocess.run(["rm", ligandFile[s]])
            elif fail > 0:
                err.info += "** Failed to fetch one of the ligands"
                for s in range(len(key_dictionary)):
                    if matchPerLigand[s] == 0:
                        subprocess.run(["rm", ligandFile[s]])
            break

    return err, matchPerLigand


####################################
#####################


def removeLine(linesToskip, inName):
    """
    Remove problematic line from pdb for pqr conversion.
    This info is gathered from the error message of pdb2pqr.

    TODO: In the case where the purged pdb file is not removed, since its output is user required,
    the final file while have those lines removed. This is not very clean and not transparent for user.
    In case a tmp file is produced for the only purpose of pqr creation, this is perfectly OK,
    the file is removed from the script at the end.
    """
    continueI = ContinueI()
    matchingLines = []
    info = []
    for line in linesToskip:
        # print(line)
        m = re.search("\s*Heavy atoms missing from ([A-Z]+)\s+([A-Z])\s+([\d]+)\\:", line)
        if m:
            # print("***"+line)
            resname = m.groups()[0]
            chain = m.groups()[1]
            resid = m.groups()[2]
            info.append({"resname": resname, "chain": chain, "resid": resid})
            matchingLines.append(
                "ATOM\s*[\d]+\s*[\S]+\s+"
                + resname
                + "\s+"
                + chain
                + "\s*"
                + resid
                + "\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)"
            )
    oldFile = open(inName, "r")
    oldData = oldFile.readlines()
    oldFile.close()
    newFile = open(inName, "w")
    for line in oldData:
        try:
            for ml in matchingLines:
                if re.match(ml, line):
                    raise continueI
        except ContinueI:
            continue  # skip the line
        newFile.write(line)

    newFile.close()
    return len(matchingLines), info


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    if n == 0:
        yield lst
        return
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def buildPQR(n, isEX, inName, savepath=".", move=False, skipLarge=False):
    err = Error()
    comment = ""
    outname = n + ".pqr"
    savepath = savepath + "/"
    if isEX:
        pdb2pqrCall = "pdb2pqr --drop-water --ff=amber --chain " + inName + " " + outname
    else:
        pdb2pqrCall = "pdb2pqr30 --drop-water --keep-chain --ff=AMBER " + inName + " " + outname
    # print(pdb2pqrCall)
    tryingToFix = True
    if isEX:
        while tryingToFix:
            try:
                out = subprocess.check_output(pdb2pqrCall, shell=True, stderr=subprocess.STDOUT)
                tryingToFix = False
            except subprocess.CalledProcessError as grepexc:
                # print ("error code", grepexc.returncode, grepexc.output)
                errlines = str(grepexc.output).split("\\n")
                # print(errlines)
                nHeavyAtomMissing, rmvdAtoms = removeLine(errlines, inName)
                # print(nHeavyAtomMissing)
                if nHeavyAtomMissing > 0:
                    tryingToFix = True
                    # try:
                    err.put_value(1)
                    err.info += (
                        "** Removed problematic heavy atom from "
                        + n
                        + ": "
                        + str(rmvdAtoms)
                        + "** "
                    )
                    comment = "\t# --> a warning was produced in pdb2pqr conversion."
                    # out=subprocess.check_output(pdb2pqrCall,shell=True,stderr=subprocess.STDOUT)

                    # except subprocess.CalledProcessError as grepexc:
                else:
                    err.put_value(2)
                    err.info += (
                        "\nCannot correct file  "
                        + n
                        + "  Unexpected exception.\n"
                        + str(grepexc.output)
                        + "\nSkipping "
                    )
                    return err, comment
            except Exception:
                print("IMPORTANT WARNING: Unexpected exception. Skipping")
                err.put_value(2)
                err.info = (
                    "Unexpected exception on  "
                    + n
                    + "\n ORIGINAL ERROR MESSAGE FROM pdb2pqr:\n"
                    + str(grepexc.output)
                    + "\n-----------"
                )
                return err, comment
    else:
        # Eror handling different for pdb2pqr30 and must be grasped from output
        while tryingToFix:
            try:
                out = subprocess.check_output(pdb2pqrCall, shell=True, stderr=subprocess.STDOUT)
                out = str(out)
                tryingToFix = False
                # print(out)
                if re.search("(CRITICAL)", out):
                    checkout = out.split("\\n")
                    nHeavyAtomMissing, rmvdAtoms = removeLine(checkout, inName)
                    # print(nHeavyAtomMissing)
                    # print(rmvdAtoms)
                    if nHeavyAtomMissing > 0:
                        # try:
                        err.put_value(1)
                        err.info += (
                            "** Removed problematic heavy atom from "
                            + n
                            + ": "
                            + str(rmvdAtoms)
                            + "** "
                        )
                        comment = "\t  # --> a warning was produced in pdb2pqr conversion."
                        tryingToFix = True
                        # out=subprocess.check_output(pdb2pqrCall,shell=True,stderr=subprocess.STDOUT)
                        # except subprocess.CalledProcessError as grepexc:
                        # err.put_value(2)
                        # err.info += "Unhandled exception on "+n+ "\n ORIGINAL ERROR MESSAGE FROM pdb2pqr:\n"+str(grepexc.output)+"\n-----------"
                    else:
                        err.put_value(2)
                        err.info += (
                            "\nCannot correct file" + n + "Unexpected exception.\n Skipping "
                        )
                        return err, comment
            except Exception:
                print("IMPORTANT WARNING: Unexpected exception. Skipping")
                err.put_value(2)
                err.info = (
                    "Unexpected exception on "
                    + n
                    + "\n ORIGINAL ERROR MESSAGE FROM pdb2pqr:\n"
                    + str(subprocess.CalledProcessError)
                    + "\n-----------"
                )
                return err, comment
    if skipLarge:
        lenght = file_len(outname)
        if lenght > 10000:
            err.put_value(3)
            err.info = (
                "Skipping "
                + n
                + " since very large (%d lines). To disable this behavior, turn off the option."
                % lenght
            )
            subprocess.run(["rm", outname])
            return err, comment
    if move:
        subprocess.run(["mv", outname, savepath])

    return err, comment


########################
############################# MAIN ################

# ERROR HANDLING:
# 2-3 print warning and causes skipping
# 1 only prints warning


# check that pdbtopqr is installed
def main(argv):
    import getopt

    onlyXYZ = False
    excludeLage = False
    isDatabase = False
    isChunk = False
    purgePDB = False
    extractPQR = False
    noTollerance = False
    verbose = True
    extension = ".pdb"
    ##OMIT?
    # skip=0

    ###
    try:
        opts, args = getopt.getopt(
            argv, "dch", ["XYZ", "excludeLarge", "PQR", "purgePDB", "safe", "quiet", "help"]
        )
    except getopt.GetoptError:
        print("uncorrect formatting of options")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            print("Usage:\npython3 lfetch\nOptions:")
            print("--XYZ: ligands extracted as coordinate files")
            print(
                "--PQR: pdb structures queried are converted to PQR. CAREFUL: needspdb2pqr intalled"
            )
            print(
                "--purgePDB: a copy of the original pdb structure without the extracted ligand is produced"
            )
            print("--safe: Partial matches to MOAD naming scheme are excluded")
            print("--quiet: No info is printed on stdout while running")
            print("-d: 'Database mode'--> all pdbs in the working folder are analysed")
            print(
                "-c: User can split the result (e.g. extracted ligands) in separate folders defining the size of each chunk"
            )
            input("\n")
            sys.exit()
        if opt in ["--excludeLarge"]:
            excludeLage = True
        if opt in ["-d"]:
            print("Database operating mode selected")
            isDatabase = True
        if opt in ["-c"]:
            if isDatabase:
                print("Chunk mode selected")
                isChunk = True
            else:
                print(
                    "inconsistency in the optional arguments given : Chunk mode must be associated to database building option -d"
                )
                sys.exit(2)
        if opt in ["--XYZ"]:
            onlyXYZ = True
            extension = ".xyz"
        if opt in ["--PQR"]:
            extractPQR = True
        if opt in ["--purgePDB"]:
            purgePDB = True
        if opt in ["--safe"]:
            noTollerance = True
        if opt in ["--quiet"]:
            verbose = False

    # uIN = input("Insert y for automatic mode. [y/n]")

    success_counter = 0
    err = Error()
    stopWatch = Crono()

    errFile = open("error_log.txt", "w")

    # Check if pdb2pqr installed and set correct path for calling
    # Tollerates both executable and python versions
    if extractPQR:
        locate = shutil.which("pdb2pqr")
        # print(locate)
        isEX = False
        if locate == None:
            # Try to find it as the pip3 version
            locate = shutil.which("pdb2pqr30")
            if locate == None:
                try:
                    raise OSError("pdb2pqr not found.\n")
                except Exception as e:
                    exit(str(e) + "EXIT")
        else:
            isEX = True

    buildDatabase = False

    ligMapFiles = []
    if isDatabase:
        # DATABASE BUILDING MODE
        buildDatabase = True

        # exclude files containing _ which are ligands or purged pdb
        infileList = [n for n in glob.glob("*.pdb") if "_" not in n]

        if not infileList:
            try:
                raise FileNotFoundError("pdb files must be placed in the running folder\n")
            except Exception as e:
                print(str(e) + "EXIT")
                err.value = 3
                err.info = str(e) + "EXIT"
        err.handle(errFile)
        if isChunk:
            nc = int(input("Insert chunk size out of " + str(len(infileList)) + "\n"))
        else:
            nc = 0
        # split pdb list into chunks
        nameList = [re.match("(.*)\.pdb", l).groups()[0] for l in infileList]
        c_infile = []
        for fc in chunks(nameList, nc):
            c_infile.append(fc)
        nc = len(c_infile)
        if not nameList:
            print("No structures to process in the working directory")
            sys.exit()

        print("Number of structures to process= ", len(nameList))
        print("number of chuncks= ", nc)
        answ = input("proceed? (y/n)\n")
        if answ == "y":
            ContinueI
        else:
            sys.exit()
    else:
        ligMapFiles.append(open("ligandMap.txt", "w"))
        ligMapFiles[0].write("# ************** PDB ligand Map *************** \n")
        ligMapFiles[0].write(
            "#\tCreated using '*BuildMap module*\n# Ligand validation based on binding MOAD database.\n"
        )
        ligMapFiles[0].write("# Author L. Gagliardi, Istituto Italiano di Tecnologia\n")
        ligMapFiles[0].write("# " + stopWatch.init())
        ligMapFiles[0].write("\n# --------------\n")
        pass

    #############  ******* CORE FUNCTIONS ******* #############################
    done = set()
    patience = False
    loopinf = True
    data = [None]
    local_path = [None]
    here = os.path.abspath(".")
    moveFile = False
    logfile = open("logfile.txt", "w")
    logfile.write("# List of succesfully processed structures")
    while loopinf:
        if not buildDatabase:
            local_path[0] = here
            try:
                pdb_name = str(input("Insert pdb name \n"))
                data[0] = [pdb_name]
            except KeyboardInterrupt:
                break
            except:
                exit("a not handled exception occurred..")
        else:
            patience = True
            loopinf = False
            moveFile = True
            data = [l for l in c_infile]
            # print(data,len(data))

            if nc > 1:
                local_path = []
                for i in range(nc):
                    local_folder = str(i + 1)
                    isFolder = os.path.isdir(here + "/" + local_folder)
                    if not isFolder:
                        subprocess.run(["mkdir", local_folder])
                    local_path.append(here + "/" + local_folder)
                    ligMapFiles.append(open(local_path[-1] + "/ligandMap.txt", "w"))
                    ligMapFiles[-1].write("# ************** PDB ligand Map *************** \n")
                    ligMapFiles[0].write(
                        "#\tCreated using '*lfetch-MOAD_ligandFinder*\n# Ligand validation is based on binding MOAD database.\n"
                    )
                    ligMapFiles[0].write("#\thttps://github.com/lucagl/MOAD_ligandFinder.git\n")
                    ligMapFiles[-1].write(
                        "# Author L. Gagliardi, Istituto Italiano di Tecnologia\n"
                    )
                    ligMapFiles[-1].write("# " + stopWatch.init())
                    ligMapFiles[-1].write("\n# --------------\n")
            else:
                local_path[0] = here + "/output"
                isFolder = os.path.isdir(here + "/output")
                if not isFolder:
                    subprocess.run(["mkdir", "output"])
                ligMapFiles.append(open("ligandMap.txt", "w"))
                ligMapFiles[0].write("# ************** PDB ligand Map *************** \n")
                ligMapFiles[0].write(
                    "#\tCreated using '*BuildMap module*\n# Ligand validation based on binding MOAD database.\n"
                )
                ligMapFiles[0].write("# Author L. Gagliardi, Istituto Italiano di Tecnologia\n")
                ligMapFiles[0].write("# " + stopWatch.init())
                ligMapFiles[0].write("\n# --------------\n")

            total = len(infileList)
            # bar = progressbar.ProgressBar(maxval=total,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            # bar.start()
        global_counter = 0

        for s, names in enumerate(data):
            logfile.write("\n\n Chunk " + str(s) + ": \n")
            for n in names:
                comment1 = ""
                comment2 = ""
                if n.lower() in done:
                    continue
                global_counter += 1

                # print('\n**'+n)
                err, ligandList = queryMOAD(n, patience)
                # print(ligandList)
                err.handle(errFile)
                if err.value == 2:
                    print("Problem: could not find ligands of " + n + " in MOAD..")
                    # ligMapFiles[s].write('\n# '+str(len(ligandList)) +"\t" + n + "\t Did not found valid ligands. IGNORING")
                    continue

                    # print(ligandList)
                stop = 0
                while stop <= 1:
                    err, matched = extractLigand(
                        n,
                        ligandList,
                        savepath=local_path[s],
                        onlyXYZ=onlyXYZ,
                        extractPQR=extractPQR,
                        purgePDB=purgePDB,
                    )
                    err.handle(errFile)
                    if err.value == 2:
                        if stop == 0:
                            stop += 1
                            print(
                                "<ERROR>: Could not extract ligands for "
                                + n
                                + ", trying to re-download structure.."
                            )
                            err.info = "Trying to re-download structure.."
                            err.value = 1
                            err.handle(errFile)
                            url = "https://files.rcsb.org/download/" + n + ".pdb"
                            print(url)
                            # print(str(["wget", url,'-O',n+".pdb"]))
                            # proc = subprocess.Popen(["wget", url,'-O',n+".pdb"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            # proc.wait()
                            # (stdout, stderr) = proc.communicate()
                            try:
                                r = requests.get(url, allow_redirects=True)
                                open(n + ".pdb", "wb").write(r.content)
                            except HTTPError:
                                # if proc.returncode != 0:
                                # print(stderr)
                                print("Could not download " + n)
                                err.value = 2
                                err.handle(errFile)
                                err.info = "Could not (re)download " + n
                                err.handle(errFile)
                                ligMapFiles[s].write(
                                    "\n# "
                                    + str(len(ligandList))
                                    + "\t"
                                    + n
                                    + "\t <-- Unable to extract ligands. Check error log. IGNORING"
                                )
                                break
                        else:
                            print("SKIPPING " + n + ". Check error log \n")
                            ligMapFiles[s].write(
                                "\n# "
                                + str(len(ligandList))
                                + "\t"
                                + n
                                + "\t <-- Unable to extract ligands. Check error log. IGNORING"
                            )
                            break
                    elif err.value == 1:
                        if not isDatabase:
                            if noTollerance:
                                print(
                                    "<ERROR> Some uncoherency between MOAD and pdb ligand name on "
                                    + n
                                    + ", check errors log..\n"
                                )
                            else:
                                print(
                                    "<WARNING> Some uncoherency between MOAD and pdb ligand name on "
                                    + n
                                    + ", check errors log..\n"
                                )
                        # comment1 = "\t\t # --> a warning was produced in ligand extractuon, this structure is less trustable..."
                        break
                    else:
                        break
                if err.value == 2:
                    continue
                if extractPQR:
                    # build pqr
                    if purgePDB:
                        inName = n + "_clean.pdb"
                    else:
                        inName = "tmp"
                    err, comment2 = buildPQR(
                        n,
                        isEX,
                        inName,
                        savepath=local_path[s],
                        move=moveFile,
                        skipLarge=excludeLage,
                    )
                    err.handle(errFile)
                else:
                    err.put_value(0)
                comment = comment1 + comment2
                # Updating ligMap
                if err.value == 3:
                    if verbose and (not isDatabase):
                        print("SKIPPING " + n + " since too large \n")
                    ligMapFiles[s].write(
                        "\n# " + str(len(ligandList)) + "\t" + n + "\t <-- Too large! IGNORING"
                    )
                    for l in ligandList:
                        lfile = local_path[s] + "/" + l["filename"] + extension
                        if os.path.isfile(lfile):
                            subprocess.run(["rm", lfile])
                        else:
                            pass

                    continue
                elif err.value == 2:
                    print("Problem: SKIPPING " + n + ". Check error log \n")
                    ligMapFiles[s].write(
                        "\n# "
                        + str(len(ligandList))
                        + "\t"
                        + n
                        + "\t Unable to extract pqr. Check error log. IGNORING"
                    )
                    continue
                else:
                    ligMapFiles[s].write(
                        "\n"
                        + str(len(ligandList) - np.sum(np.logical_not(matched)))
                        + "\t"
                        + n
                        + comment
                    )
                    logfile.write("\n" + n)
                    for k, d in enumerate(ligandList):
                        if matched[k] == 1:
                            ligMapFiles[s].write("\n" + d["filename"])
                        elif matched[k] == -1:
                            if noTollerance:
                                lfile = local_path[s] + "/" + d["filename"] + extension
                                subprocess.run(["rm", lfile])
                                ligMapFiles[s].write(
                                    "\n#"
                                    + d["filename"]
                                    + "\t# <-- Partial match with respect to what expected. SAFE MODE=skipping"
                                )
                            else:
                                ligMapFiles[s].write(
                                    "\n"
                                    + d["filename"]
                                    + "\t# <-- Partial match with respect to what expected.."
                                )
                        else:
                            ligMapFiles[s].write("\n#" + d["filename"] + " <-- NOT found in pdb")

                errFile.flush()
                ligMapFiles[s].flush()
                done.add(n.lower())
                if verbose:
                    if isDatabase:
                        # print("processed: " +n)
                        print("\r%d / %d of structures processed" % (global_counter, total), end="")
                    else:
                        print("Valid ligands found: ", len(ligandList))
                        if noTollerance:
                            print("Excluded:", np.sum(matched == 0) + np.sum(matched == -1))
                            print(
                                "Ligands:",
                                [
                                    l["filename"]
                                    for l, m in zip(ligandList, matched)
                                    if (m != -1) and (m != 0)
                                ],
                            )
                        else:
                            print("Excluded:", np.sum(matched == 0))
                            print(
                                "Ligands:",
                                [l["filename"] for l, m in zip(ligandList, matched) if m != 0],
                            )
                #     bar.update(global_counter)
                success_counter += 1
    # if buildDatabase:
    #     bar.finish()

    errFile.close()

    if extractPQR and data:
        if not purgePDB:
            subprocess.run(["rm", "tmp"])

    if extractPQR:
        if not isEX:
            # remove <pdbName>.log files generated by pdb2pqr30
            subprocess.run("rm *.log", shell=True)
    for lgmf in ligMapFiles:
        lgmf.close()

    print("TOTAL NUMBER OF STRUCTURES SUCCESFULLY PROCESSED = " + str(success_counter))
    logfile.write(
        "\n\n** TOTAL NUMBER OF STRUCTURES SUCCESFULLY PROCESSED = " + str(success_counter)
    )
    logfile.close()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print("\nUser exit")
        sys.exit()
