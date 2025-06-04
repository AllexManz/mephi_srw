import json
import os
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from typing import Dict, List

app = Flask(__name__)

class SplunkIntegration:
    def __init__(self, model_endpoint: str, splunk_config: Dict):
        self.model_endpoint = model_endpoint
        self.splunk_config = splunk_config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {splunk_config["api_key"]}',
            'Content-Type': 'application/json'
        })

    def _transform_to_prompt(self, event: Dict) -> str:
        """Convert Splunk JSON event to text prompt"""
        base_template = """Security event analysis request:
        Timestamp: {timestamp}
        Source: {source}
        Event Type: {event_type}
        Details: {message}

        Additional context:
        - Criticality: {criticality}
        - Related assets: {assets}
        - Previous similar events: {similar_events}

        Please analyze this event and provide:
        1. Potential threat classification
        2. Recommended actions
        3. Priority level"""
        
        return base_template.format(
            timestamp=datetime.fromisoformat(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            source=event.get('source', 'unknown'),
            event_type=event.get('event_type', 'unknown'),
            message=self._format_message(event),
            criticality=event.get('criticality', 'medium'),
            assets=', '.join(event.get('assets', [])),
            similar_events=event.get('similar_events_count', 0)
        )

    def _format_message(self, event: Dict) -> str:
        """Format event-specific message"""
        if event['event_type'] == 'brute_force_attempt':
            return (f"Multiple login attempts from {event['src_ip']} to {event['dst_ip']} "
                   f"({event['attempts']} attempts detected)")
        elif event['event_type'] == 'malware_alert':
            return f"Malware detected: {event['malware_name']} on {event['host']}"
        else:
            return str(event.get('message', 'No details available'))

    def process_event(self, event: Dict) -> Dict:
        """Main processing pipeline"""
        try:
            # Step 1: Transform to prompt
            prompt = self._transform_to_prompt(event)
            
            # Step 2: Send to model
            model_response = self._query_model(prompt)
            
            # Step 3: Parse response
            return self._parse_model_response(model_response)
        except Exception as e:
            return {
                'error': str(e),
                'original_event': event
            }

    def _query_model(self, prompt: str) -> Dict:
        """Send prompt to language model"""
        response = self.session.post(
            self.model_endpoint,
            json={'prompt': prompt, 'max_tokens': 500},
            timeout=5
        )
        response.raise_for_status()
        return response.json()

    def _parse_model_response(self, response: Dict) -> Dict:
        """Convert model response to structured format"""
        try:
            content = response['choices'][0]['text']
            return {
                'analysis': self._extract_section(content, 'Analysis'),
                'recommendations': self._extract_list(content, 'Recommended actions'),
                'priority': self._extract_priority(content),
                'raw_response': content
            }
        except (KeyError, IndexError):
            return {'error': 'Invalid model response format'}

    def _extract_section(self, text: str, title: str) -> str:
        """Extract section between titles"""
        start_idx = text.find(title)
        if start_idx == -1:
            return ''
        start_idx += len(title) + 1  # Skip title and newline
        end_idx = text.find('\n\n', start_idx)
        return text[start_idx:end_idx].strip()

    def _extract_list(self, text: str, title: str) -> List[str]:
        """Extract bullet list items"""
        section = self._extract_section(text, title)
        return [line[2:].strip() for line in section.split('\n') if line.startswith('- ')]

    def _extract_priority(self, text: str) -> str:
        """Extract priority level"""
        section = self._extract_section(text, 'Priority level')
        return section.lower().replace('priority:', '').strip()

# Initialize the integration service
config = {
    'api_key': os.getenv('SPLUNK_API_KEY', 'your_splunk_api_key'),
    'model_endpoint': os.getenv('MODEL_ENDPOINT', 'http://model-service:5000/predict')
}

integrator = SplunkIntegration(config['model_endpoint'], config)

@app.route('/splunk/webhook', methods=['POST'])
def handle_splunk_event():
    """Endpoint for receiving events from Splunk"""
    try:
        event = request.json
        
        # Validate required fields
        if not event or 'timestamp' not in event:
            return jsonify({'error': 'Invalid event format'}), 400
        
        # Process the event
        result = integrator.process_event(event)
        
        # Return the analysis result to Splunk
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == "__main__":
    # Run the Flask server
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('DEBUG', 'False').lower() == 'true')