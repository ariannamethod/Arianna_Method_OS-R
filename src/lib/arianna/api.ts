export interface AriannaResult {
  reply: string;
}

export async function sendAriannaMessage(message: string, endpoint = '/arianna'): Promise<AriannaResult> {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message})
  });

  if(!response.ok) {
    throw new Error(`Arianna request failed with status ${response.status}`);
  }

  return response.json() as Promise<AriannaResult>;
}
