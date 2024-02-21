from Service import Service

class SpeechRecognitionService (Service):   
    def __init__(self, memory_srv, speech_srv, params={}):
        Service.__init__(self)
        self.memory_srv = memory_srv
        self.speech_srv = speech_srv

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

    def onWordRecognized(self, value):
        if value[1] > self.threshold :
            print ("you say : '{}' ({})".format(value[0], value[1]))
        # if(len(value) > 1 and value[1] >= self.threshold):
        #     print(value[0])

    def stop(self):
        self.speech_srv.unsubscribe(self.user)
        print ('Speech recognition engine stopped for user {}'.format(self.user))