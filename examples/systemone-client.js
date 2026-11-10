// This is a simple smoke-test client for systemone-api.
//
// Usage:
//
// 1. Start systemone-api.
// 2. Run: `node examples/systemone-client.js`

const HOST = process.env.HOST ? process.env.HOST : '127.0.0.1';
const PORT = process.env.PORT ? Number(process.env.PORT) : 9990;

async function main() {
    const body = {
        state: {
            customer_message: 'I was charged twice for my subscription.'
        },
        questions: {
            department: {
                type: 'choice',
                instructions: 'Choose the correct support department.',
                criteria: {
                    billing: 'Payments, invoices, refunds, or duplicate charges',
                    shipping: 'Delivery status or damaged shipments',
                    technical: 'Software bugs or account access problems'
                }
            }
        },
        none_check: false,
        return_details: true
    };

    const response = await fetch(`http://${HOST}:${PORT}/v1/systemone`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    });
    const json = await response.json();
    console.log(JSON.stringify(json, null, 2));

    const tokensPerSecond = json.usage.input_tokens / (json.timing_ms.first_pass_sum / 1000);
    console.log(`Input throughput: ${tokensPerSecond.toFixed(2)} tok/s`);

    if (!response.ok) {
        throw new Error(`SystemOne API returned HTTP ${response.status}`);
    }
    if (json.answers?.department?.choice !== 'billing') {
        throw new Error('Expected the billing department');
    }
}

main().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
