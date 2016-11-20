

class qasm_loader:
	"""
	QASM File Loder
	"""
	def __init__(self, file_name):
		print("[+] pyqx : qasm_loader : loading file '%s' ..." % file_name)
		self.file_name = file_name
		with open(file_name) as f:
			self.lines = f.readlines()
		# lines pre-processing
		for i in range(0,len(self.lines)):
			self.lines[i] = self.lines[i].replace("\n","")  # remove empty lines
			c = self.lines[i].find("#")			   # remove comments
			if c != -1:
				self.lines[i] = self.lines[:c]

	def load_circuits(self):
		n = len(self.lines)
		# print("[+] lines : ", n)
		i = 0
		self.circuits = []
		
		# load circuits
		while i < n:
			l = self.lines[i]
			i = i + 1
			if len(l) == 0:
				continue
			if l[0] != '.':
				continue
			# process a new subcircuit
			l = l.replace(" ","")
			print("[+] new circuit : ",l);
			cn = l[1:]  # circuit name
			# cr = ""     # circuit
			cr = []     # circuit
			# p = [cn,""] 
			p = [cn,[]] 
			l = self.lines[i]
			end = False
			ng = 0  # number of gates
			while (i < n) and (end == False):
				l = self.lines[i]
				if l == []:       
					i = i+1
					continue
				if l == "":       # skip empty lines
					i = i+1
					continue
				if (l[0] == '.'):
					end = True;
				else:
					if (len(l) > 1):
						# cr = cr + l + "; "
						cr.append(l)
						ng = ng + 1
					i = i + 1
			p[1] = cr
			print("[-] number of gates : ",ng)
			self.circuits.append(p)
	
	def get_circuits(self):
		return self.circuits

'''

# loading qasm test

ql = qasm_loader('/Users/nader/Develop/demo/rb.qc')
ql.load_circuits()
circuits = ql.get_circuits()
for i in range(0,len(circuits)):
	print(circuits[i])
	print("")
'''

