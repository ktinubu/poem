import grpc
import poetry_pb2 
import poetry_pb2_grpc
import poetry_gen
from concurrent import futures
import grpc

if __name__ == '__main__':
	gen = poetry_gen.FullModel()
	print(gen.predict("poop"))

class GeneratePoetryServicer(poetry_pb2_grpc.GeneratePoetryServicer):
	"""Provides methods that implement functionality of GeneratePoetry server."""
	def __init__(self):
		self.full_model = poetry_gen.FullModel()

	def GeneratePoetry(self, request, context):
		return self.full_model.predict(request.Text)

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  poetry_pb2_grpc.add_GeneratePoetryServicer_to_server(
      GeneratePoetryServicer(), server)
  server.add_insecure_port('localhost:8080')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  serve()