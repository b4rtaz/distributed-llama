const { Socket } = require('net');

// Run Distributed Llama server:
//
// `./dllama simple-server --weights-float-type q40 --buffer-float-type q80 --nthreads 4 --model converter/dllama_meta-llama-3-8b_q40.bin --tokenizer converter/llama3-tokenizer.t --workers 10.0.0.1:9999`
//
// Then run this script:
//
// `node examples/nodejs-example.cjs`

class DistributedLlamaClient {
	current = null;
	bufferSize = null;
	buffer = null;
	queue = [];

	constructor(host, port) {
		this.socket = new Socket();
		this.socket.connect(port, host, () => {
			this.socket.on('data', this.onData);
			this.socket.on('error', this.onError);
			this.next();
		});
	}

	onData = (payload) => {
		if (this.bufferSize === null) {
			this.bufferSize = payload.readInt32LE(0);
			this.buffer = payload.subarray(4);
		} else {
			this.buffer = Buffer.concat([this.buffer, payload]);
		}
		if (this.buffer.length === this.bufferSize) {
			const value = this.buffer.toString('utf-8');
			this.current.resolve(value);
			this.current = null;
			this.bufferSize = null;
			this.buffer = null;
			this.next();
		}
	};

	onError = (err) => {
		if (this.current) {
			this.current.reject(err);
			this.current = null;
		}
	};

	next() {
		if (this.current || this.queue.length === 0) {
			return;
		}
		this.current = this.queue.pop();

		const promptBytes = Buffer.from(this.current.prompt);
		const payloadBytes = Buffer.alloc(8 + promptBytes.length);
		payloadBytes.writeInt32LE(promptBytes.length, 0);
		payloadBytes.writeInt32LE(this.current.maxTokens, 4);
		payloadBytes.fill(promptBytes, 8);
		this.socket.write(payloadBytes, error => {
			if (error && this.current) {
				this.current.reject(error);
				this.current = null;
			}
		});
	}

	generate(prompt, maxTokens = 128) {
		return new Promise((resolve, reject) => {
			this.queue.push({
				prompt,
				maxTokens,
				resolve,
				reject
			});
			this.next();
		});
	}
	
	close() {
		this.socket.end();
	}
}

async function main() {
	try {
		const client = new DistributedLlamaClient('127.0.0.1', 9990);

		const prompt0 = 'The answer to the universe really is';
		const response0 = await client.generate(prompt0);
		console.log({ prompt0, response0 });

		const prompt1 = '1 + 3 is ';
		const response1 = await client.generate(prompt1);
		console.log({ prompt1, response1 });

		client.close();
	} catch (e) {
		console.error(e);
	}
}

main();
