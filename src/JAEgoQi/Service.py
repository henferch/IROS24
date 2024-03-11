class Service:
    def __init__(self, params):
        self.parameters = params
        self.user = params['expID']
        self.realRobot = self.parameters['robotSource'] == 'robot' # ["robot", "sim"]
    def step(self):
        return None
    def stop(self):
        return None
    def sendData(self):
        return None