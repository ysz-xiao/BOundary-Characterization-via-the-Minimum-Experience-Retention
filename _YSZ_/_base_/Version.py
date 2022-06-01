class Version():
    def __init__(self, A, B, C):
        super(Version, self).__init__()
        self.version_A = A
        self.version_B = B
        self.version_C = C

    def print_version(self):
        print("YSZ version {}.{}.{}".format(self.version_A,self.version_B,self.version_C))
