from PyQt5.QtWidgets import *
import random


class InputDialog(QDialog):
    def __init__(self, input_list, column_mapping_dict, parent=None):
        super().__init__(parent)
        self.input_list = input_list
        self.input_map = column_mapping_dict
        layout = QFormLayout(self)

        self.labels = dict()
        for input in input_list:
            inputBox = QLineEdit(self)
            if(input in column_mapping_dict.keys()):
                availableInputs = list(column_mapping_dict[input].keys())
                inputBox.setText(random.choice(availableInputs))
            elif(input.lower() == "year" ):
                inputBox.setText(str(random.randint(1950, 2021)))
            elif(input.lower() == "odometer"):
                inputBox.setText(str(random.randint(0, 400000)))

            layout.addRow(input, inputBox)
            self.labels[input] = inputBox

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self);
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputList(self):
        inputs = list()
        for input in self.input_list:
            inputbox = self.labels[input]
            gvnValue = inputbox.text()
            if(input in self.input_map.keys()):
                inputDict = self.input_map[input]
                gvnNumber = 0
                for valueName in inputDict.keys():
                    if(valueName == gvnValue):
                        gvnNumber = int(inputDict[valueName])
                        break
            else:
                gvnNumber = int(gvnValue)
            inputs.append(gvnNumber)
        return inputs

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    dialog = InputDialog()
    if dialog.exec():
        print(dialog.getInputs())
    exit(0)