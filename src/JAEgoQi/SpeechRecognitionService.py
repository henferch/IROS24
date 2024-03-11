from Service import Service

class SpeechRecognitionService (Service):   
    def __init__(self, memory_srv, speech_srv, params={}):
        Service.__init__(self, params)
        self.memory_srv = memory_srv
        self.speech_srv = speech_srv

        self.realRobot = self.parameters['robotSource'] == 'robot' # ["robot", "sim"]

        self.word = None

        if self.realRobot:
            # put the service in pause for configuration
            self.speech_srv.pause(True)
            self.speech_srv.setLanguage("English")

            self.speech_srv.setVisualExpression(True)
            self.speech_srv.setParameter("Sensitivity", params['micSensitivity'])

            vocabulary = params['vocabulary']
            enableWordSpotting = False
            self.speech_srv.setVocabulary( vocabulary, enableWordSpotting )

            # resume the service 
            self.speech_srv.pause(False)     
            
            # Start the speech recognition engine with user Test_ASR
            print ('Speech recognition engine started for user {}'.format(self.user))
            self.speech_srv.subscribe(self.user)

            self.subscriber = self.memory_srv.subscriber("WordRecognized")
            self.subscriber.signal.connect(self.onWordRecognized)

            # confidence threshold
            self.threshold = 0.3
        else:
            self.simulation = self.parameters["simulation"]["speech"]

    def onWordRecognized(self, value):
        if value[1] > self.threshold :
            self.word = value[0]
            print ("you saied : '{}' ({})".format(self.word, value[1]))
        # if(len(value) > 1 and value[1] >= self.threshold):
        #     print(value[0])

    def step(self, t):
        if not self.realRobot:
            for i in range(len(self.simulation)):
                s = self.simulation[i] 
                if s['t'] <= t:
                    self.word = s['w']
                    self.simulation.pop(i)
                    break
        word = self.word
        self.word = None
        return word

    def stop(self):
        if self.realRobot:
            self.speech_srv.unsubscribe(self.user)
        print ('Speech recognition engine stopped for user {}'.format(self.user))